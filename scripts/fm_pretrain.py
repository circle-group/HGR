# fm_pretrain.py 

import sys, os

# os.environ["WANDB_MODE"] = "online"
# os.environ["_DEBUG_"] = "False" # 设置是否打印调试信息

from hgr.utils.loader import set_env_from_config
set_env_from_config()
import wandb
import argparse
import torch
torch.set_float32_matmul_precision("high") 
# torch.backends.cudnn.benchmark = True  # 对卷积模型额外有益
# torch.backends.cuda.enable_cudnn_sdp(False)


from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from datetime import datetime
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import GradScaler
from torch_geometric.utils import to_dense_batch
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR


from hgr.utils.file_utils import PathManager
from hgr.utils.loader import load_config, init_exp, load_device, load_seed
from hgr.utils.debug_utils import _DEBUG_, make_wandb_init_config, update_wandb_config
from hgr.foundation.data_utils.mol_dataset import load_pretrain_dataset
from hgr.foundation.models.encoder import GNN
from hgr.foundation.models.ema import EMA
from hgr.foundation.models.sigreg_loss import SIGRegLoss
from hgr.foundation.fm_utils import get_adamw_param_groups

from hgr.foundation.models.grammar_encoder import RuleTransformerEncoder 
# from hgr.foundation._ablation_models.graphormer import GraphormerEncoder as RuleTransformerEncoder # Graphormer-style for ablation study
# from hgr.foundation._ablation_models.pureGNN import PureGNNEncoder as RuleTransformerEncoder # PureGNN-style for ablation study
# from hgr.foundation._ablation_models.grammarGPS import GrammarAttnEncoder as RuleTransformerEncoder
import logging
logger = logging.getLogger(__name__)




def _get_amp_and_precision(cfg):
    amp_cfg = getattr(cfg, "amp", {})

    # ======== TF32 & matmul precision ========
    if torch.cuda.is_available() and amp_cfg.get("tf32", True):
        # ✅ 启用 TF32 加速
        torch.backends.cuda.matmul.allow_tf32 = True # 控制 矩阵乘法 (matmul) 是否允许 TF32
        torch.backends.cudnn.allow_tf32 = True # 控制 卷积 (cudnn) 是否允许 TF32

    # ======== AMP dtype & scaler ========
    # 如果配置里没写，默认开启 bf16（最稳且普适的加速）
    enabled = amp_cfg.get("enabled", True)
    dtype_str = amp_cfg.get("dtype", "bf16")

    print(f"AMP enabled: {enabled}, dtype: {dtype_str}")
    if dtype_str.lower() in ('bf16', 'bfloat16'):
        return enabled, torch.bfloat16, False  # bf16 不需要 scaler
    elif dtype_str.lower() in ('fp16', 'float16', 'half'):
        return enabled, torch.float16, True
    else:
        # 其他（如 'fp32'）相当于禁用 AMP
        return False, torch.float32, False




def infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch




# ---------- Loss 构建  ---------- #
def build_criteria(cfg, loss_fn="sce", alpha_l=1.0, device=None):

    # 1. node-level
    # SCE: (1 - cos_sim)^alpha 的均值
    def sce_loss(x, y, alpha=1.0):
        cos = F.cosine_similarity(x.float(), y.float(), dim=-1, eps=1e-8)
        return (1.0 - cos).pow(alpha).mean()

    if 'node' in cfg.pretexts:
        criterion_node = partial(sce_loss, alpha=alpha_l) if loss_fn == "sce" else nn.CrossEntropyLoss()
    else:
        criterion_node = None

    criterion_subg = nn.BCEWithLogitsLoss() if 'subgraph' in cfg.pretexts else None

    # 3. graph-level
    criterion_desc = nn.MSELoss() if 'descriptors' in cfg.pretexts else None

    # 4. 3D-level（上三角 MSE，diagonal=1 默认不含主对角）
    def criterion_dist_fn(node_emb, pos, node2graph, diagonal=1):
        """
        返回：上三角（diagonal=1不含对角；0含对角）位置上的 MSE 均值
        """
        # 将变长图打包成 (B, Lmax, D) + mask 
        # 找到批次中节点数最多的图的节点数（假设为 Lmax），然后将所有图都填充（pad）到这个长度。
        X, _ = to_dense_batch(node_emb, node2graph)           # (B, L, D), (B, L)
        P, pmask = to_dense_batch(pos, node2graph)        # (B, L, 3), (B, L)

        # 显式到 FP32，copy=False 避免不必要拷贝
        X = X.to(torch.float32, copy=False)
        P = P.to(torch.float32, copy=False)

        # 一次性按图计算 (B, L, L) 距离矩阵
        Dpred = torch.cdist(X, X)                                 # (B, L, L)
        with torch.no_grad():
            Dtrue = torch.cdist(P, P)                             # (B, L, L)

        B, L, _ = Dpred.shape
        tri = torch.triu(torch.ones(L, L, device=X.device, dtype=torch.bool), diagonal=diagonal)  # (L, L)
        tri = tri.unsqueeze(0) & pmask.unsqueeze(2) & pmask.unsqueeze(1)   # (B, L, L) 只选择真实的节点对（忽略填充部分

        mse = (Dpred - Dtrue).square() 
        return mse[tri].mean()

    criterion_dist = criterion_dist_fn if 'conf' in cfg.pretexts else None

    # === 5. SigReg ===
    criterion_sigreg = SIGRegLoss(num_slices=cfg.tasks.sigreg.num_slices).to(device) if 'sigreg' in cfg.pretexts else None

    return criterion_node, criterion_subg, criterion_desc, criterion_dist, criterion_sigreg


def forward_and_loss(batch, model_list, pretexts, criterions, weights, loss_fn="sce",
                     amp_enabled=True, amp_dtype=torch.bfloat16, device=None, device_type="cuda", is_train=False, global_step=None):
    """
    统一的前向 + 损失计算逻辑。
    支持训练与评估两种模式。
    返回 total_loss, task_losses(dict)
    """
    encoder, dec_atoms, dec_subg, dec_descriptor, dec_dist = model_list
    criterion_node, criterion_subg, criterion_des, criterion_dist, criterion_sigreg = criterions

    with torch.set_grad_enabled(is_train), torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=amp_enabled):
        node_rep, graph_rep = encoder(batch)
        pred_node       = dec_atoms(node_rep, batch.edge_index, batch.edge_attr, batch.masked_atom_indices) if 'node' in pretexts else None
        pred_subg       = dec_subg(graph_rep) if 'subgraph' in pretexts else None
        pred_descriptor = dec_descriptor(graph_rep) if 'descriptors' in pretexts else None
        node_emb        = dec_dist(node_rep) if 'conf' in pretexts else None

    with torch.autocast(device_type=device_type, enabled=False):
        total = torch.zeros((), device=device, dtype=torch.float32)
        task_losses = {}

        if 'node' in pretexts:
            if loss_fn == "sce":
                loss_node = criterion_node(batch.node_attr_label.float(),
                                           pred_node[batch.masked_atom_indices].float())
            else:
                loss_node = criterion_node(pred_node[batch.masked_atom_indices].float(),
                                           batch.mask_node_label[:, 0]).float()
            total += weights['node'] * loss_node
            task_losses['node'] = loss_node

        if 'subgraph' in pretexts:
            loss_subg = criterion_subg(pred_subg.float(), batch.fingerprint.float())
            total += weights['subgraph'] * loss_subg
            task_losses['subgraph'] = loss_subg

        if 'descriptors' in pretexts:
            loss_desc = criterion_des(pred_descriptor.float(), batch.descriptors_ss.float())
            total += weights['descriptors'] * loss_desc
            task_losses['descriptors'] = loss_desc

        if 'conf' in pretexts:
            loss_dist = criterion_dist(node_emb, batch.pos, batch.batch)
            total += weights['conf'] * loss_dist
            task_losses['conf'] = loss_dist

        if 'sigreg' in pretexts and is_train:
            loss_sigreg = criterion_sigreg(graph_rep.float())
            total += weights['sigreg'] * loss_sigreg
            task_losses['sigreg'] = loss_sigreg

    return total, task_losses


@torch.no_grad()
def evaluate_epoch(val_loader, model_list, pretexts, criterions, weights, loss_fn="sce", 
                device=None, device_type="cuda", amp_enabled=True, amp_dtype=torch.bfloat16):
    """
    仅用于验证/评估的 epoch。
    返回两个 dict:
      - stats: 各任务未加权平均 loss
      - stats_w: 各任务加权平均 loss (乘 λ)
    """
    for model in model_list:
        if model is not None:
            model.eval()

    # 初始化统计量
    stats = {f"val/{k}": 0.0 for k in ['total'] + pretexts}
    stats_w = {f"val_w/{k}": 0.0 for k in pretexts}

    n_batches = 0
    for batch in tqdm(val_loader, desc="eval", disable=_DEBUG_, leave=False):
        batch = batch.to(device, non_blocking=True)
        total, task_losses = forward_and_loss(
            batch, model_list, pretexts, criterions, weights,
            loss_fn=loss_fn,
            amp_enabled=amp_enabled, amp_dtype=amp_dtype,
            device=device, device_type=device_type,
            is_train=False, global_step=None
        )

        n_batches += 1
        stats["val/total"] += float(total.item())
        for k, v in task_losses.items():
            stats[f"val/{k}"] += float(v.item())
            stats_w[f"val_w/{k}"] += float((weights[k] * v).item())

    # ===== 平均化 =====
    for k in stats:
        stats[k] /= max(1, n_batches)
    for k in stats_w:
        stats_w[k] /= max(1, n_batches)

    # 合并返回
    return {**stats, **stats_w}





def train_by_steps(cfg, model_list, opt_list, schedulers, ema,
                   train_loader, val_loader, device,
                   loss_fn="sce",
                   amp_enabled=True, amp_dtype=torch.bfloat16, scaler=None,
                   log_steps=100, eval_every=2000, save_every=2000,
                   output_dir=None, model_tag=None):
    """
    固定 total_steps 训练；每 eval_every steps 在 EMA 上做一次全量验证。
    带 tqdm 进度条和详细 val 打印。
    """
    # encoder, dec_atoms, dec_subg, dec_descriptor, dec_dist = model_list
    criterions = build_criteria(cfg, loss_fn=loss_fn, device=device)
    pretexts = cfg.pretexts
    device_type = getattr(device, "type", "cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(amp_enabled and device_type == "cuda")

    clip_params = [p for m in model_list if m is not None for p in m.parameters() if p.requires_grad]
    # Train mode
    for m in model_list:
        if m is not None: m.train()

    global_step = 0
    best_val = float('inf')
    losses_log, losses_w_log = {k: 0.0 for k in pretexts}, {k: 0.0 for k in pretexts}
    weights = {k: float(getattr(cfg.tasks[k], "lambda", 0.0)) for k in pretexts}
    
    
    train_iter = infinite_loader(train_loader)
    pbar = tqdm(total=cfg.train.total_steps, desc="Training", dynamic_ncols=True)
    while global_step < cfg.train.total_steps:
        batch = next(train_iter).to(device, non_blocking=True)
        for opt in opt_list:
            if opt is not None:
                opt.zero_grad(set_to_none=True)


        total, task_losses = forward_and_loss(batch, model_list, pretexts, criterions, weights, loss_fn=loss_fn, 
                                              amp_enabled=use_amp, amp_dtype=amp_dtype, device=device, device_type=device_type,
                                              is_train=True, global_step=global_step)
                
        # ===== 反向 + 更新 =====
        if scaler is not None and scaler.is_enabled(): # fp16 路径, 实际代码中走的是bf16
            scaler.scale(total).backward()

            # 1) 对每个优化器先 unscale，确保所有 param.grad 都回到真实量级
            for opt in opt_list:
                if opt is not None: scaler.unscale_(opt)

            clip_grad_norm_(clip_params, max_norm=1.0)
            for opt in opt_list:
                if opt is not None: scaler.step(opt)
            scaler.update()
        else:
            total.backward()
            clip_grad_norm_(clip_params, max_norm=1.0)
            for opt in opt_list:
                if opt is not None: opt.step()

        for sch in schedulers: sch.step()
        if ema is not None: ema.update(model_list[0])

        

        # ==== 累加 ====
        for k, v in task_losses.items():
            losses_log[k] += float(v.item())
            losses_w_log[k] += float((weights[k] * v).item())

        # ===== 日志与进度 =====
        global_step += 1
        pbar.update(1)
        if global_step % log_steps == 0:
            cur_lr = schedulers[0].get_last_lr()[0] if schedulers else opt_list[0].param_groups[0]['lr']
            pbar.set_postfix({"step_loss": f"{total.item():.4f}", "lr": f"{cur_lr:.2e}"})

            log_dict = {"train/step_loss": float(total.item()), "global_step": global_step, "learning_rate": cur_lr}
            # 遍历所有 loss 类型
            for k in losses_log:
                log_dict[f"train/{k}_loss"] = losses_log[k]/log_steps
                log_dict[f"train_w/{k}_loss"] = losses_w_log[k]/log_steps
            
            wandb.log(log_dict, step=global_step)    
            losses_log, losses_w_log = {k: 0.0 for k in pretexts}, {k: 0.0 for k in pretexts} # 清空losses_log和losses_w_log


        # ===== 验证 =====
        total_val = float('inf')
        if (global_step > 0) and (global_step % eval_every == 0):
            with ema.average_parameters(model_list[0]):
                val_stats = evaluate_epoch(val_loader, model_list, pretexts, criterions, weights,
                    loss_fn=loss_fn, device=device, device_type=device_type, amp_enabled=use_amp, amp_dtype=amp_dtype)
            wandb.log(val_stats, step=global_step)

            # 验证后切回训练模式
            for m in model_list:
                if m is not None: m.train()

             # === 控制台打印验证结果 ===
            total_val = val_stats.get("val/total", 0.0)
            tqdm.write(f"[Eval @ step {global_step}] total={total_val:.6f} | "
                       f"node={val_stats.get('val/node', 0.0):.4f} (λ {val_stats.get('val_w/node', 0.0):.4f}) | "
                       f"subg={val_stats.get('val/subgraph', 0.0):.4f} (λ {val_stats.get('val_w/subgraph', 0.0):.4f}) | "
                       f"desc={val_stats.get('val/descriptors', 0.0):.4f} (λ {val_stats.get('val_w/descriptors', 0.0):.4f}) | "
                       f"conf={val_stats.get('val/conf', 0.0):.4f} (λ {val_stats.get('val_w/conf', 0.0):.4f})")

        # === 保存 checkpoint ===
        if output_dir and ((global_step % save_every == 0) or (total_val < best_val)):
            best_val = min(best_val, total_val)
            ckpt = {"encoder": model_list[0].state_dict(),
                "ema": ema.state_dict(),
                "config": cfg,
                "global_step": global_step}
            ckpt_path = os.path.join(output_dir, f"{cfg.data.name}_step{global_step}_ValLoss_{total_val:.6f}_{model_tag}.pth")
            torch.save(ckpt, ckpt_path)
            
            

        # # 清理缓存
        # if global_step % 2000 == 0:
        #     torch.cuda.empty_cache()

    pbar.close()





def build_model_opt(cfg, train_loader, device, verbose=False):
    # 参数解析
    modalities = set(cfg.modalities)
    pretexts = set(cfg.pretexts)
    
    if 'grammar' in modalities:
        # encoder = GraphGrammarEncoder(cfg.graph_encoder, cfg.grammar).to(device)
        encoder = RuleTransformerEncoder(cfg.grammar).to(device)
    else:
        encoder = GNN(cfg.graph_encoder).to(device)


    opt_encoder = torch.optim.AdamW(
        get_adamw_param_groups(encoder, cfg.train.wd),
        lr=cfg.train.lr,
        fused=True if torch.cuda.is_available() else False,
    )

    
    dec_atoms, dec_subg, dec_descriptor, dec_dist = None, None, None, None
    opt_atoms, opt_subg, opt_descriptor, opt_dist = None, None, None, None
    emb_dim = cfg.tasks.emb_dim

    # Node-level decoder
    if 'node' in pretexts:
        from hgr.foundation.models.decoder import GNNDecoders
        node_cfg = cfg.tasks.node
        dec_atoms = GNNDecoders(emb_dim, node_cfg).to(device)
        opt_atoms = torch.optim.AdamW(get_adamw_param_groups(dec_atoms, node_cfg.wd),
                                     lr=node_cfg.lr, fused=True if torch.cuda.is_available() else False)

    # Subgraph-level (MACCS & Morgan)
    if 'subgraph' in pretexts:
        from hgr.foundation.models.decoder import SubgraphPredictor
        subg_cfg = cfg.tasks.subgraph
        subg_cfg.out_dim = train_loader.dataset.data['fingerprint'].shape[1] 
        dec_subg = SubgraphPredictor(emb_dim, subg_cfg).to(device)
        opt_subg = torch.optim.AdamW(get_adamw_param_groups(dec_subg, subg_cfg.wd),
                                     lr=subg_cfg.lr, fused=True if torch.cuda.is_available() else False)

    # Graph-level (Descriptor)
    if 'descriptors' in pretexts:
        from hgr.foundation.models.decoder import DescriptorPredictor
        desc_cfg = cfg.tasks.descriptors
        desc_cfg.out_dim = train_loader.dataset.data['descriptors_ss'].shape[1] # des_dim
        dec_descriptor = DescriptorPredictor(emb_dim, desc_cfg).to(device)
        opt_descriptor = torch.optim.AdamW(get_adamw_param_groups(dec_descriptor, desc_cfg.wd),
                                           lr=desc_cfg.lr, fused=True if torch.cuda.is_available() else False)

    # 3D-level
    if 'conf' in pretexts:
        from hgr.foundation.models.decoder import Decoder_DistPreds
        conf_cfg = cfg.tasks.conf
        dec_dist = Decoder_DistPreds(emb_dim, conf_cfg).to(device)
        opt_dist = torch.optim.AdamW(get_adamw_param_groups(dec_dist, conf_cfg.wd),
                                     lr=conf_cfg.lr, fused=True if torch.cuda.is_available() else False)


    model_list = [encoder, dec_atoms, dec_subg, dec_descriptor, dec_dist]
    opt_list = [opt_encoder, opt_atoms, opt_subg, opt_descriptor, opt_dist]

    if verbose:
        print('-'*30 + ' [Model Architecture] ' + '-'*30)
        print("### set up models, one for pre-training and one for context embeddings")
        for idx, m in enumerate(model_list):
            print(f'model {idx + 1}')
            print(m)
        print()

    return model_list, opt_list


def main(cfg):
    load_seed(cfg.seed)
    device = load_device()
    
    modalities = set(cfg.modalities)
    pretexts = set(cfg.pretexts)

    train_loader, val_loader = load_pretrain_dataset(cfg, modalities, pretexts, device)

    # Encoder
    if 'grammar' in modalities:
        if hasattr(cfg.grammar, 'vocab_size'):
            assert cfg.grammar.vocab_size == train_loader.dataset.rule_vocab_size, "Vocab size mismatch"
        else:
            cfg.grammar.vocab_size = train_loader.dataset.rule_vocab_size
        logger.info(f"Vocab size: {cfg.grammar.vocab_size}")


    model_list, opt_list = build_model_opt(cfg, train_loader, device, verbose=True)
    ema = EMA(model_list[0], decay=0.999, use_num_updates=True, fp32_shadow=True)

    # Warmup and Cosine Decay
    total_steps = cfg.train.total_steps
    warmup_steps = int(getattr(cfg.train, "warmup_ratio", 0.05) * total_steps)  # 比如 5%

    # Scheduler
    schedulers = []
    floor_ratio = 0.1
    for opt in opt_list:
        if opt is None:  continue
        base_lr = opt.param_groups[0]['lr']
        s1 = LinearLR(opt, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps)
        s2 = CosineAnnealingLR(opt, T_max=total_steps - warmup_steps, eta_min=floor_ratio * base_lr)
        schedulers.append(SequentialLR(opt, schedulers=[s1, s2], milestones=[warmup_steps]))


    # 模型标签
    # model_tag = f"Abl-Graphormer-NoBias-D{cfg.tasks.emb_dim}L{cfg.grammar.num_layers},bs{cfg.train.batch_size},lr{cfg.train.lr},wd{cfg.train.wd},dp{cfg.grammar.dropout}"
    # model_tag = 'onlyGrammar'
    model_tag = 'HGR'
    print(f"@@ Model tag: {model_tag}")

    # 输出目录
    now_md = datetime.now().strftime("%m%d")
    output_dir = os.path.join(PathManager.CKPT_DIR, f"{model_tag}_{now_md}") # 输出目录
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {os.path.abspath(output_dir)}")
    print(f"Input Modalities: {modalities}, Pretexts: {pretexts}")

    amp_enabled, amp_dtype, need_scaler = _get_amp_and_precision(cfg)
    scaler = GradScaler(enabled=need_scaler)


    # ==== Steps-based training  ====
    eval_every = getattr(cfg.train, "eval_every_steps", 2000)
    save_every = getattr(cfg.train, "save_every_steps", 2000)

    train_by_steps(cfg, model_list, opt_list, schedulers, ema,
                   train_loader, val_loader, device,
                   loss_fn=cfg.train.loss_fn,
                   amp_enabled=amp_enabled, amp_dtype=amp_dtype, scaler=scaler,
                   log_steps=100, eval_every=eval_every, save_every=save_every,
                   output_dir=output_dir, model_tag=model_tag)
        
        
    

if __name__ == '__main__':
    # 1) 环境 & 配置
    # setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, default='configs/pretrain.yaml',
        help='YAML path relative to the repository root (configs/...) or an absolute path',
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    env_cfg = init_exp(cfg)


    wandb.init(
        entity=env_cfg.wandb.entity,  # team name
        project="HoGen-pure-TF",  # project name
        name=f"{os.getenv('WANDB_SWEEP', '')}{cfg.data.name}-{cfg.exp_time}",  # experiment name
        dir=PathManager.WANDB_DIR,
        config=make_wandb_init_config(cfg),
        allow_val_change=True,  # 允许在调参时修改配置
    )
    wandb.run.log_code(
        root="FM",
        include_fn=lambda path: path.endswith(".py")
    ) # 记录代码

    cfg = update_wandb_config(cfg)
    main(cfg)

    wandb.finish()
    print("Finished FM/pretrain.py")
