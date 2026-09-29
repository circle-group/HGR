# fm_finetune.py

import os, sys, warnings
from hgr.utils.loader import set_env_from_config
from hgr.utils.file_utils import resolve_ckpt_path
set_env_from_config()
warnings.filterwarnings('ignore')  # keep behavior


import random
import textwrap
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from easydict import EasyDict as edict

# ✅ 启用 TF32 加速
torch.backends.cuda.matmul.allow_tf32 = True # 控制 矩阵乘法 (matmul) 是否允许 TF32
torch.backends.cudnn.allow_tf32 = True # 控制 卷积 (cudnn) 是否允许 TF32
torch.backends.cudnn.benchmark = True  # 对卷积模型额外有益

from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error  # for regression
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingWarmRestarts, SequentialLR, CosineAnnealingLR

from hgr.utils.debug_utils import Timer
from hgr.utils.loader import load_seed, load_config, init_exp, load_device
from hgr.foundation.models.ema import EMA, NoOpEMA
from hgr.foundation.models.finetune_models import EncoderProbe
from hgr.foundation.data_utils.mol_dataset import load_finetune_dataset
from hgr.foundation.data_utils.dataloader import FinetuneDataLoader
from hgr.foundation.data_utils.mol_defs import DATASET_NUM_TASKS
from hgr.foundation.fm_utils import get_adamw_param_groups

import logging
logger = logging.getLogger(__name__)




# =========================
# Training / Evaluation
# =========================

def train(model, device, loader, optimizer, scheduler, ema):
    """Binary/multi-label classification training. """
    model.train()
    total_loss = 0.0  # [EQ-OPT] 累加平均损失
    criterion = nn.BCEWithLogitsLoss(reduction="none") # 二分类交叉熵损失（带 logit 输入），但这里 reduction="none" → 不自动求平均/求和，而是返回逐样本逐任务的损失矩阵。

    for batch in tqdm(loader, desc="Iteration", leave=False):
        batch = batch.to(device, non_blocking=True)
        for opt in optimizer:
            if opt is not None: opt.zero_grad(set_to_none=True)
        pred, sigreg_loss = model(batch, sigreg=True)
        y = batch.y.view(pred.shape)#.to(torch.float64)
    
        # Whether y is non-null or not.数据集中有时会用 0 表示“缺失标签”（因为是多任务，某些任务可能没标注），所以只在 y != 0 的地方计算 loss。
        is_valid = y.ne(0) # 与 y**2>0 等价，但更快且避免不必要的幂运算。
        # Loss matrix 逐元素的损失矩阵
        loss_mat = criterion(pred, (y + 1) / 2)  # {-1,1}->{0,1}, 
        loss_mat = loss_mat.masked_fill(~is_valid, 0) #loss matrix after removing null target  
        loss = torch.sum(loss_mat)/torch.sum(is_valid) # 只计算非零标签的平均损失
        # if sigreg_loss is not None:
        #     loss += 0.05 * sigreg_loss

        loss.backward()
        if not getattr(model, "freeze_encoder", False):
            clip_grad_norm_(model.encoder.parameters(), max_norm=1.0)
        clip_grad_norm_(model.probe.parameters(),   max_norm=1.0)

        for opt, sch in zip(optimizer, scheduler):
            if opt is not None: opt.step()
            if sch is not None: sch.step()
        if not getattr(model, "freeze_encoder", False):
            ema.update(model.encoder)

        total_loss += loss.item()

    avg_train_loss = total_loss /len(loader)
    return avg_train_loss



def train_reg(args, model, device, loader, optimizer, scheduler, ema):
    """Regression training. 保持原始范数与归一化策略不变。"""
    model.train() # 自定义了train, freeze_encoder=True时model.encoder.eval()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Iteration", leave=False):
        batch = batch.to(device, non_blocking=True)
        pred = model(batch)[0]
        y = batch.y.view(pred.shape) #.to(torch.float64)

        if args.data.name in ['qm7', 'qm8', 'qm9']:
            loss = torch.sum(torch.abs(pred - y)) / y.size(0)
        elif args.data.name in ['esol', 'freesolv', 'lipophilicity']:
            loss = torch.sum((pred - y) ** 2) / y.size(0)

        for opt in optimizer: 
            if opt is not None: opt.zero_grad(set_to_none=True)
        loss.backward()
        if not getattr(model, "freeze_encoder", False):
            clip_grad_norm_(model.encoder.parameters(), max_norm=1.0)
        clip_grad_norm_(model.probe.parameters(),   max_norm=1.0)
        for opt, sch in zip(optimizer, scheduler):
            if opt is not None: opt.step()
            if sch is not None: sch.step()
        if not getattr(model, "freeze_encoder", False):
            ema.update(model.encoder)

        total_loss += loss.item()

    avg_train_loss = total_loss / len(loader)
    return avg_train_loss



@torch.no_grad()
def eval(model, device, loader):
    model.eval()
    y_true, y_scores = [], []

    for batch in tqdm(loader, desc="Iteration", leave=False):
        batch = batch.to(device, non_blocking=True)
        pred = model(batch)[0]

        y_true.append(batch.y.view(pred.shape).cpu())
        y_scores.append(pred.cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_scores = torch.cat(y_scores, dim=0).numpy()

    roc_list = []
    for i in range(y_true.shape[1]):
        # AUC 仅在该任务既有正样本又有负样本时定义 # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == -1) > 0:
            is_valid = (y_true[:,i]**2 > 0)
            roc_list.append(roc_auc_score((y_true[is_valid, i] + 1) / 2, y_scores[is_valid, i]))

    # if len(roc_list) < y_true.shape[1]:
    #     logger.warning(f"Some target is missing, mising ratio {1 - float(len(roc_list)) / y_true.shape[1]}")

    return sum(roc_list) / len(roc_list)

@torch.no_grad()
def eval_reg(args, model, device, loader):
    model.eval()
    y_true, y_scores = [], []

    for batch in tqdm(loader, desc='Iteration', leave=False):
        batch = batch.to(device, non_blocking=True)
        pred = model(batch)[0]
        y_true.append(batch.y.view(pred.shape).cpu())
        y_scores.append(pred.cpu())

    y_true = torch.cat(y_true, dim=0).numpy().flatten()
    y_scores = torch.cat(y_scores, dim=0).numpy().flatten()

    mse = mean_squared_error(y_true, y_scores)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_scores)
    return mse, mae, rmse




def build_data_loaders(cfg, datasets, device, run_seed):
    train_ds, val_ds, test_ds = datasets

    def seed_worker(worker_id):
        # 基于 run_seed 派生每个 worker 的子种子，保证多 worker 也可复现
        base = (run_seed if run_seed is not None else 0)
        worker_seed = (base + worker_id) % (2**32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    num_workers = cfg.train.num_workers_loader

    loader_kwargs = {
        'batch_size': cfg.train.batch_size,
        'modalities': cfg.modalities,
        'max_path_distance': cfg.train.max_path_distance,
        'num_workers': num_workers,
        'pin_memory': (device.type == 'cuda'),
        'persistent_workers': num_workers > 0,
    }

    # 让 DataLoader 的 shuffle 明确受 run_seed 控制
    g = torch.Generator()
    g.manual_seed(run_seed if run_seed is not None else 0)

    train_loader = FinetuneDataLoader(train_ds, shuffle=True,
        # drop_last=False,          # 训练建议丢最后小 batch，稳定 BN/梯度统计
        generator=g, 
        worker_init_fn=seed_worker if num_workers > 0 else None,
        **loader_kwargs)
    val_loader = FinetuneDataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = FinetuneDataLoader(test_ds, shuffle=False, **loader_kwargs)

    loaders = [train_loader, val_loader, test_loader]
    return loaders

def shutdown_loaders(loaders):
    """Ensure DataLoader workers are torn down promptly to avoid long waits between runs."""
    for loader in loaders:
        iterator = getattr(loader, "_iterator", None)
        if iterator is not None:
            try:
                iterator._shutdown_workers()
            except Exception:
                pass  # best-effort cleanup
    # dereference to help GC
    del loaders[:]

def load_model(cfg, num_tasks, device):
    """初始化模型与优化器。"""

    cfg_probe = edict({'num_tasks': num_tasks})
    
    if cfg.pretrain_ckpt != "" and cfg.train.ft_type != 'no_pretrain':
        ckpt_path = resolve_ckpt_path(cfg.pretrain_ckpt)

        ckpt = torch.load(ckpt_path, map_location='cpu') # 先加载到 CPU，然后移动到目标设备，避免设备不匹配问题
        
        # 同步 config 中的参数
        loaded_cfg = edict(ckpt['config'])
        cfg.grammar.emb_dim = loaded_cfg.grammar.emb_dim
        cfg.grammar.num_layers = loaded_cfg.grammar.num_layers
        if 'rule_emb_dim' in loaded_cfg.grammar:
            cfg.grammar.rule_emb_dim = loaded_cfg.grammar.rule_emb_dim
        # cfg.grammar.rule_emb_mode = loaded_cfg.grammar.rule_emb_mode
        # cfg.graph_encoder.num_layers = loaded_cfg.graph_encoder.num_layers

        model = EncoderProbe(cfg.grammar, cfg_probe)
        model.from_pretrained(ckpt)
        print(f'Loaded pretrained checkpoint from {os.path.abspath(ckpt_path)}')
    else:
        model = EncoderProbe(cfg.grammar, cfg_probe)
        print(f"No pretrained model loaded.")

    # 组装优化器参数
    if cfg.train.ft_type == 'freeze':
        # freeze 模式冻结 enocder 参数
        for p in model.encoder.parameters():
            p.requires_grad_(False)
        model.freeze_encoder = True
    else:
        model.freeze_encoder = False
            


    model.to(device)
    return model

def build_optimizer_scheduler(cfg, model, steps_per_epoch):
    ft_type = cfg.train.ft_type

    # lr 配置：probe 大，encoder 小（10×差距起步）
    probe_lr = cfg.train.lr 
    enc_lr   = cfg.train.lr * getattr(cfg.train, "enc_lr_ratio", 0.1)    
    opt_enc   = optim.AdamW(get_adamw_param_groups(model.encoder, cfg.train.decay), lr=enc_lr,   betas=(0.9, 0.999)) if ft_type != 'freeze' else None
    opt_probe = optim.AdamW(get_adamw_param_groups(model.probe,   cfg.train.decay), lr=probe_lr, betas=(0.9, 0.999))
    optimizer = [opt_enc, opt_probe]

    # === 调度器 ===
    # steps_per_epoch = len(loaders[0])
    total_steps     = steps_per_epoch * cfg.train.epochs
    warmup_steps    = int(getattr(cfg.train, "warmup_ratio", 0.1) * total_steps) # 默认0.2

    # # probe：与你现在类似，1 epoch 重启 + 高 floor
    # s1p = LinearLR(opt_probe, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)
    # s2p = CosineAnnealingWarmRestarts(opt_probe, T_0=steps_per_epoch, T_mult=1, eta_min=getattr(cfg.train, "probe_eta_min", 0.30) * probe_lr)
    # sched_probe = SequentialLR(opt_probe, [s1p, s2p], milestones=[warmup_steps])

    # # encoder：更慢的周期 + 更低 floor（或不重启）
    # s1e = LinearLR(opt_enc, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)
    # s2e = CosineAnnealingWarmRestarts(opt_enc, T_0=3*steps_per_epoch, T_mult=2, # T0=3* (之前默认2，5也不好), T_mult=2
    #                                 eta_min=0.05 * enc_lr) # 默认=0.05, (0.001, 0.1, 0.2都不如0.05)
    # sched_enc = SequentialLR(opt_enc, [s1e, s2e], milestones=[warmup_steps])
    # scheduler = [sched_enc, sched_probe]

    s1p = LinearLR(opt_probe, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps)
    # s2p = CosineAnnealingWarmRestarts(opt_probe, T_0=steps_per_epoch, T_mult=1,  eta_min=getattr(cfg.train, "probe_eta_min", 0.30) * probe_lr) # 默认0.3
    s2p = CosineAnnealingLR(opt_probe, T_max=total_steps - warmup_steps, eta_min=getattr(cfg.train, "probe_eta_min", 0.1) * probe_lr)
    sched_probe = SequentialLR(opt_probe, [s1p, s2p], milestones=[warmup_steps])

    
    sched_enc = None
    if ft_type != 'freeze':
        s1e = LinearLR(opt_enc, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps)
        # s2e = CosineAnnealingWarmRestarts(opt_enc, T_0=getattr(cfg.train, "encoder_T_0", 3)*steps_per_epoch, 
        #                                 T_mult=2, eta_min=getattr(cfg.train, "encoder_eta_min", 0.05) * enc_lr) # 默认0.05
        s2e = CosineAnnealingLR(opt_enc, T_max=total_steps - warmup_steps, eta_min=getattr(cfg.train, "enc_eta_min", 0.1) * enc_lr)
        sched_enc = SequentialLR(opt_enc, [s1e, s2e], milestones=[warmup_steps])
    scheduler = [sched_enc, sched_probe]

    return optimizer, scheduler
    


def run_epoch(args, task_type, model, device, loaders, optimizer, scheduler, ema, eval_train=True):
    """跑 1 个 epoch：训练 + 评估，返回 (train_metric, val_metric, test_metric)。"""
    train_loader, val_loader, test_loader = loaders

    # ===== 训练 =====
    if task_type == 'cls':
        train_loss = train(model, device, train_loader, optimizer, scheduler, ema)
    else:
        train_loss = train_reg(args, model, device, train_loader, optimizer, scheduler, ema)

    # ===== 验证/测试都用 EMA 权重 =====
    with ema.average_parameters(model.encoder):
        if task_type == 'cls':
            train_acc = eval(model, device, train_loader) if eval_train else 0.0
            val_acc   = eval(model, device, val_loader)
            test_acc  = eval(model, device, test_loader)
            return train_loss, train_acc, val_acc, test_acc
        else:
            if eval_train:
                _, train_mae, train_rmse = eval_reg(args, model, device, train_loader)
            else:
                train_mae, train_rmse = 0, 0
            _, val_mae,  val_rmse  = eval_reg(args, model, device, val_loader)
            _, test_mae, test_rmse = eval_reg(args, model, device, test_loader)
            if args.data.name in ['esol', 'freesolv', 'lipophilicity']:
                return train_loss, train_rmse, val_rmse, test_rmse
            else:
                return train_loss, train_mae,  val_mae,  test_mae



def main(cfg):

    device = load_device() 

    # 任务类型与任务数
    task_type = 'cls' if cfg.data.name in ['tox21','hiv','pcba','muv','bace','bbbp','toxcast','sider','clintox','mutag'] else 'reg'  
    if task_type == 'reg':
        if cfg.data.name in ['esol', 'freesolv', 'lipophilicity']:
            metric_name = 'RMSE'
        elif cfg.data.name in ['qm7', 'qm8', 'qm9']:
            metric_name = 'MAE'
    elif task_type == 'cls':
        metric_name = 'AUC'
    
    num_tasks = DATASET_NUM_TASKS[cfg.data.name]
    ft_type = cfg.train.ft_type
    patience = getattr(cfg.train, "patience", 8)

    # # Dataset
    train_ds, val_ds, test_ds = load_finetune_dataset(cfg)
    if 'grammar' in cfg.modalities:
        cfg.grammar.vocab_size = train_ds.rule_vocab_size
    
    

    # 汇总容器
    train_his, val_his, test_his = [], [], []
    for run_idx, seed in enumerate(cfg.seed_list):
        run_seed = None if (seed is None or seed == -1) else int(seed)
        if run_seed is not None:
            load_seed(run_seed)
        
        
        # —— 每个 seed：重建模型与优化器（受 run seed 影响） ——
        loaders = build_data_loaders(cfg, (train_ds, val_ds, test_ds), device, run_seed=run_seed)
        model = load_model(cfg, num_tasks, device)
        # if run_idx == 0:
        #     print('-'*30 + '[model & optimizer]' + '-'*30)
        #     print(model)
        #     print(optimizer)

        ema_decay = getattr(cfg.train, "ema_decay", 0.0)
        if ema_decay <=0.0 or ft_type == 'freeze':
            ema = NoOpEMA() 
        else:
            ema = EMA(model.encoder, decay=ema_decay, use_num_updates=True, fp32_shadow=True)


        optimizer, scheduler = build_optimizer_scheduler(cfg, model, steps_per_epoch = len(loaders[0]))
        
        print(f'================== {cfg.data.name} Run {run_idx+1}: {cfg.train.ft_type}, data_seed={cfg.data_seed}, run_seed={seed} ==================')
        best_tuple = None    # (train, val, test, epoch)
        patience_counter = 0
        for epoch in range(1, cfg.train.epochs + 1):
            best_tag = ''
            train_loss, tr, va, te = run_epoch(cfg, task_type, model, device, loaders, optimizer, scheduler, ema, eval_train=cfg.train.eval_train)
            if epoch >= 5:
                if best_tuple is None or (task_type == 'cls' and va > best_tuple[1]) or (task_type == 'reg' and va < best_tuple[1]):
                    best_tuple = (tr, va, te, epoch)
                    best_tag = '*'
                    patience_counter = 0 # 如果性能提升，重置计数器
                else:
                    patience_counter += 1 # 如果性能没有提升，增加计数器
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch}")
                        break
                    
            print(f"Epoch {epoch} | train loss: {train_loss:.6f} | {metric_name}: train: {tr:.6f}, val: {va:.6f}, test: {te:.6f}{best_tag}")

        train_rel, val_rel, test_rel, best_epoch = best_tuple
        print(f"[BEST - Epoch: {best_epoch}] train: {train_rel:.6f} val: {val_rel:.6f} test: {test_rel:.6f}")

        # 分类以 *100 存储
        if task_type == 'cls':
            train_rel, val_rel, test_rel = train_rel*100, val_rel*100, test_rel*100

        shutdown_loaders(loaders) # 及时关闭 DataLoader worker，避免在多 seed 时上一轮的进程拖延退出
        train_his.append(train_rel)
        val_his.append(val_rel)
        test_his.append(test_rel)
        print()
        
        

    # 汇总
    train_his, val_his, test_his = map(np.asarray, (train_his, val_his, test_his))
    title = f"[{cfg.data.name}] {len(cfg.seed_list)} runs summary ({cfg.train.ft_type})"
    print('='*30 + f'{title:^30}' + '='*30)
    msg = f"Ckpt:{cfg.pretrain_ckpt}"
    print("\n".join(textwrap.wrap(msg, width=90)))
    print('@ test_results:', test_his)
    if task_type == 'cls':
        print(f"@ mean±std: train: {train_his.mean():.1f}±{train_his.std(ddof=1):.2f}, "
              f"val: {val_his.mean():.1f}±{val_his.std(ddof=1):.2f}, "
              f"test: {test_his.mean():.1f}±{test_his.std(ddof=1):.2f}")
    else:
        print(f"@ mean±std: train: {train_his.mean():.6f}±{train_his.std(ddof=1):.6f}, "
              f"val: {val_his.mean():.6f}±{val_his.std(ddof=1):.6f}, "
              f"test: {test_his.mean():.6f}±{test_his.std(ddof=1):.6f}")
    print('='*90)

    return test_his.mean()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, default='configs/finetune.yaml',
        help='YAML path relative to the repository root (configs/...) or an absolute path',
    )
    # 新增：批量脚本会传入的三个参数
    parser.add_argument('--override_data_name', type=str, default=None,
                        help="覆盖 cfg.data.name，例如 bbbp/tox21/toxcast")
    parser.add_argument('--override_pretrain_ckpt', type=str, default=None,
                        help="覆盖 cfg.pretrain_ckpt，传入具体 .pth 路径")
    parser.add_argument('--save_log_file', type=str, default=None,
                        help="若提供该路径，则所有 print 输出仅写入该文件；未提供则打印到控制台")
    parser.add_argument('--only_seed', type=int, default=None,
                    help="若指定，仅运行该 seed（覆盖 cfg.seed_list）")
    parser.add_argument('--override_pretrain_grammar_path', type=str, default=None,
                        help="覆盖 cfg.grammar.grammar_path，传入具体 .pklz 路径")
    parser.add_argument('--override_batch_size', type=int, default=None)
    parser.add_argument(
        '--ft-type', choices=('full', 'freeze', 'no_pretrain'), default=None,
        help='Override train.ft_type from the config',
    )
    args = parser.parse_args()
    cfg = load_config(args.config)


    # 应用覆盖（若提供）
    if args.override_data_name is not None:
        cfg.data.name = args.override_data_name
    if args.override_pretrain_ckpt is not None:
        cfg.pretrain_ckpt = args.override_pretrain_ckpt
    if args.override_pretrain_grammar_path is not None:
        cfg.grammar.pretrain_grammar_path = args.override_pretrain_grammar_path
    if args.only_seed is not None:
        cfg.seed_list = [int(args.only_seed)]
    if args.override_batch_size is not None:
        cfg.train.batch_size = args.override_batch_size
    if args.ft_type is not None:
        cfg.train.ft_type = args.ft_type

    env_cfg = init_exp(cfg)

    # ========= 日志重定向：到文件 or 控制台 =========
    _orig_stdout = sys.stdout
    _log_fp = None
    if args.save_log_file is not None:
        log_dir = os.path.dirname(args.save_log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        _log_fp = open(args.save_log_file, "a", encoding="utf-8")
        sys.stdout = _log_fp  # 只写文件，不再打印到控制台

    # ========= 执行 finetune =========
    with Timer('Finetune Time', enabled=True):
        test_mean = main(cfg)  # 保持：main(cfg) 返回 test_his.mean()

    # 末尾追加一行，供 batch 正则解析
    print(f"[INFO] Finetune finished, mean test metric: {test_mean:.4f}")

    # ========= 恢复 stdout =========
    if _log_fp is not None:
        _log_fp.flush()
        _log_fp.close()
        sys.stdout = _orig_stdout
