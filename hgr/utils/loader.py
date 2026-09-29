import os
import math
import torch
import socket
import random
import yaml
import logging
import numpy as np
from datetime import datetime
from easydict import EasyDict as edict
from contextlib import contextmanager
from hgr.utils.file_utils import PathManager, get_runtime_paths
from hgr.utils.config import resolve_config_path
from hgr.utils.debug_utils import setup_logging

logger = logging.getLogger(__name__)

try:
    import pynvml
except ImportError:
    pynvml = None





def init_exp(config, env_cfg_path='env_config.yaml'):
    setup_logging()

    if 'PYTHONHASHSEED' not in os.environ:
        logger.info("Please set PYTHONHASHSEED to speed up evaluation.")
    else:
        logger.info((f"PYTHONHASHSEED={os.environ['PYTHONHASHSEED']}"))

    env_cfg = set_env_from_config(path=os.path.join(PathManager.CFG_ROOT, env_cfg_path))
    PathManager.init(data_name=config.data.name, exp_name=config.exp_name)

    return env_cfg


def set_env_from_config(path="configs/env_config.yaml"):
    """Load an optional machine-local environment profile.

    Public runs do not require this file: runtime paths are derived from
    ``ASSET_ROOT`` and W&B may infer the entity or read ``WANDB_ENTITY``.
    Internal deployments may keep host-specific profiles in ``env_config.yaml``.
    """
    config = {}
    if path and os.path.isfile(path):
        logger.info(f"Setting environment variables from {path} ...")
        with open(path, encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Environment config must contain a YAML mapping: {path}")
        config = loaded
    else:
        logger.info("Optional environment config not found; using environment variables and defaults.")

    default_config = config.setdefault("default", {})
    if not isinstance(default_config, dict):
        raise ValueError("Environment config section 'default' must be a YAML mapping")
    wandb_config = config.setdefault("wandb", {})
    if not isinstance(wandb_config, dict):
        raise ValueError("Environment config section 'wandb' must be a YAML mapping")
    if not wandb_config.get("entity"):
        wandb_config["entity"] = os.environ.get("WANDB_ENTITY")

    hostname = socket.gethostname()
    prefix = hostname.split('.')[0]

    # 1. 应用 default 段
    def set_if_missing(name, value):
        if name in os.environ:
            return
        if value is None or isinstance(value, (dict, list)):
            raise ValueError(
                f"Environment variable {name!r} must have a scalar, non-null value"
            )
        expanded = os.path.expanduser(os.path.expandvars(str(value)))
        os.environ[name] = expanded

    for k, v in default_config.items():
        if k in os.environ: # 如果当前环境里已有该变量值，就跳过
            continue
        set_if_missing(k, v)

    # 2. 应用 host-specific 段
    for key, envs in config.items():
        if key in {"default", "wandb"}:
            continue
        if not isinstance(envs, dict):
            continue
        host_prefixes = envs.get("HOST_PREFIXES", [])
        if isinstance(host_prefixes, str):
            host_prefixes = [host_prefixes]

        host_candidates = [key, *host_prefixes]
        if any(prefix.startswith(candidate) for candidate in host_candidates):
            logger.info(f"Matched env profile '{key}' for host '{prefix}'")
            for k, v in envs.items():
                if k in os.environ or k == "HOST_PREFIXES":
                    continue # 只要环境里已有，就不覆盖
                set_if_missing(k, v)
            break

    runtime_paths = get_runtime_paths()
    for k, v in runtime_paths.items():
        os.environ.setdefault(k, v)
    return edict(config)

# ─── Config Loading ───────────────────────────────────────────────────────────
def load_config(config_path):
    config_path = resolve_config_path(config_path, repo_root=PathManager.PRO_ROOT)
    #config = edict(yaml.load(open(config_path, 'r', encoding='utf-8'), Loader=yaml.FullLoader))
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    def _auto_cast(d):
        """
        递归遍历字典，先展开环境变量，再把字符串形式的数字转换成 int 或 float
        """
        if isinstance(d, dict): return {k: _auto_cast(v) for k, v in d.items()}
        elif isinstance(d, list): return [_auto_cast(v) for v in d]
        elif isinstance(d, str):
            d = os.path.expanduser(os.path.expandvars(d))
            # 尝试转 int
            try:
                if '.' not in d and 'e' not in d and 'E' not in d:
                    return int(d)
            except ValueError:
                pass
            # 尝试转 float
            try:
                return float(d)
            except ValueError:
                return d
        else:
            return d

    config = edict(_auto_cast(config))
    config.exp_time = datetime.now().strftime('%y%m%d-%H%M%S')
    return config



def load_seed(seed):
    # Set random seed
    # os.environ['PYTHONHASHSEED'] = str(123)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return seed

def load_device():
    usage_type = os.getenv('USAGE_TYPE', 'memory')
    def get_device_usage(handle, usage_type):
        """
        获取 GPU 设备的指定使用情况。
        :param handle: GPU 句柄
        :param usage_type: 'GPU' 表示获取利用率，'memory' 表示获取内存使用情况
        :return: 对应的使用情况数值
        """
        assert usage_type in ['mix', 'GPU', 'memory']
        if usage_type == 'GPU':
            return pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        elif usage_type == 'memory':
            return pynvml.nvmlDeviceGetMemoryInfo(handle).used
        elif usage_type == 'mix':
            used_memory = pynvml.nvmlDeviceGetMemoryInfo(handle).used
            total_memory = pynvml.nvmlDeviceGetMemoryInfo(handle).total
            gpu_rate = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu / 100
            temperature = (pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU) - 30)/40
            return used_memory/total_memory + gpu_rate + temperature

    if torch.cuda.is_available():
        if pynvml is None:
            logger.warning("pynvml is not installed; falling back to the first visible CUDA device.")
            print(f"device available (physical GPU ids):{os.getenv('CUDA_VISIBLE_DEVICES', '0')}")
            print("use device: 0 (index 0)")
            return torch.device('cuda:0')

        pynvml.nvmlInit()
        visible_devices_env = os.getenv("CUDA_VISIBLE_DEVICES")
        # MIG UUIDs (e.g. "MIG-0fa7467c-...") aren't integer-indexable via pynvml;
        # fall back to using the first visible device directly.
        if visible_devices_env is not None and "MIG-" in visible_devices_env:
            print(f"device available (MIG UUIDs):{visible_devices_env}")
            print("use device: cuda:0 (first visible MIG slice)")
            pynvml.nvmlShutdown()
            return torch.device("cuda:0")

        if visible_devices_env is not None:
            visible_devices = [int(dev) for dev in visible_devices_env.split(",")]
        else:
            visible_devices = list(range(pynvml.nvmlDeviceGetCount()))
        print(f"device available (physical GPU ids):{visible_devices}")

        # Single visible device → skip pynvml scoring (also avoids errors on MIG-enabled cards)
        if len(visible_devices) == 1:
            print(f"use device: {visible_devices[0]} (index 0, single visible device)")
            pynvml.nvmlShutdown()
            return torch.device("cuda:0")

        used_list = []
        for i in visible_devices:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                usage = get_device_usage(handle, usage_type)
            except pynvml.NVMLError as exc:
                logger.warning("pynvml query failed for GPU %d (%s); assuming max usage.", i, exc)
                usage = float("inf")
            used_list.append(usage)

        # If all pynvml queries failed, just pick cuda:0
        if all(u == float("inf") for u in used_list):
            logger.warning("All pynvml queries failed; defaulting to cuda:0")
            pynvml.nvmlShutdown()
            return torch.device("cuda:0")

        min_index = np.argmin(used_list)
        selected_gpu = visible_devices[min_index]
        print(f"use device: {selected_gpu} (index {min_index})")
        device = torch.device(f'cuda:{min_index}')

        pynvml.nvmlShutdown()

    else:
        print("CUDA not available!")
        device = torch.device('cpu')

    return device



# ─── Optimizer + Scheduler Wrappers ─────────────────────────────────────────
def load_optimizer(cfg, params):
    """根据 config 生成一个 Adam optimizer"""
    opt_name = cfg.train.optimizer.lower()
    if opt_name == 'adam':
        optimizer =  torch.optim.Adam(params, lr=cfg.train.lr,)
    elif opt_name == 'adamw':
        optimizer =  torch.optim.AdamW(
            params,
            lr=cfg.train.lr,
            betas=(0.9, 0.999),
            eps=cfg.train.eps,
            weight_decay=cfg.train.weight_decay
        )
    else:
        raise NotImplementedError(f"Unsupported optimizer: {opt_name}")


    if cfg.train.lr_decay < 1.0:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.train.lr_decay)
    else:
        scheduler = None
    return optimizer, scheduler

def optimization_manager(config):
    """ warmup + grad_clip """
    def optimize_fn(optimizer, params, step):
        # 1) lr warmup
        if config.train.warmup > 0:
            # warmup：学习率预热. 在训练初期，学习率从 0 平滑地增加到设定的 lr 值，避免模型在一开始就梯度爆炸或发散。
            lr_scale = min(step / config.train.warmup, 1.0)
            for g in optimizer.param_groups:
                g['lr'] = config.train.lr * lr_scale

        # 2) Gradient Clipping
        if config.train.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=config.train.grad_clip)

    return optimize_fn





@contextmanager
def seed_context(seed):
    """
    PYTHONHASHSEED 必须在 Python 解释器启动前生效。一旦进入代码运行阶段，哈希种子已经固定，在脚本中通过 os.environ 修改它不会改变当前进程中 hash() 的行为。
    """
    # 保存当前状态
    # old_pythonhashseed = os.environ.get('PYTHONHASHSEED', None)
    torch_state = torch.get_rng_state()
    torch_cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    numpy_state = np.random.get_state()
    random_state = random.getstate()

    # 设置新种子
    # if old_pythonhashseed is not None:
    #     os.environ['PYTHONHASHSEED'] = old_pythonhashseed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        yield
    finally:
        # 恢复原状态
        # if old_pythonhashseed is not None:
        #     os.environ['PYTHONHASHSEED'] = old_pythonhashseed
        # else:
        #     os.environ.pop('PYTHONHASHSEED', None)

        torch.set_rng_state(torch_state)
        if torch_cuda_state is not None:
            torch.cuda.set_rng_state_all(torch_cuda_state)
        np.random.set_state(numpy_state)
        random.setstate(random_state)
        # 还原 cudnn 设置
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True





class EarlyStopping:
    """
    多指标 EarlyStopping。
    可以监控多项指标（如 loss, NSPDK, FCD, validity），并在指定逻辑下提前停止训练。

    Args:
        metrics_cfg (dict): 每个要监控指标的配置字典，结构如下：
            {
                "loss":     {"mode": "min", "patience": 5, "min_delta": 1e-4},
                "NSPDK":    {"mode": "min", "patience": 8, "min_delta": 1e-5},
                "FCD":      {"mode": "min", "patience": 8, "min_delta": 1e-3},
                "validity": {"mode": "max", "patience": 10, "min_delta": 1e-3},
                # 如果某指标使用同一套 patience/min_delta 也可以只给出 mode，如：
                # "another_metric": {"mode": "max"}  # 会使用默认的 patience/min_delta
            }
        trigger_mode (str): 'all' 或 'any'。
            'all'：当所有指标都超过 patience 时，才触发提前停止。
            'any'：只要任意一个指标超过 patience，就触发提前停止。

        default_patience (int): 若某个指标在 metrics_cfg 中没有指定 patience， 则使用此默认值。默认为 10。
        default_min_delta (float): 若某个指标未指定 min_delta，则使用此默认值。默认为 0.0。
        verbose (bool): 是否在 step() 时打印每个指标的状态。
    """

    def __init__(self,
                 metrics_cfg: dict,
                 start_epoch: 30,
                 trigger_mode: str = 'all',
                 default_patience: int = 10,
                 default_min_delta: float = 1e-5,
                 verbose: bool = True):
        assert trigger_mode in ('any', 'all'), "trigger_mode must be 'any' or 'all'"

        self.trigger_mode = trigger_mode
        self.verbose = verbose
        self.start_epoch = start_epoch # 训练达到一定轮数后才开始考虑 early stopping

        # 内部字典：每个指标对应的设置
        # key: metric name, value: dict(mode, patience, min_delta)
        self.metrics_cfg = {}

        # 对每个指标，记录它的 best_value、num_bad_epochs
        self.best_value = {}
        self.num_bad_epochs = {}

        # 将传入的 metrics_cfg 进行“补全”：若某个指标没指定 patience / min_delta，就用默认值
        for metric, cfg in metrics_cfg.items():
            mode = cfg.get('mode', 'min')
            patience = cfg.get('patience', default_patience)
            min_delta = cfg.get('min_delta', default_min_delta)
            assert mode in ('min', 'max'), f"Metric {metric} mode must be 'min' or 'max'"

            self.metrics_cfg[metric] = {'mode': mode, 'patience': patience, 'min_delta': min_delta}

            # 初始化 best_value
            self.best_value[metric] = float('inf') if mode == 'min' else float('-inf')
            # 初始化连续“坏 epoch”计数
            self.num_bad_epochs[metric] = 0
        logger.info(f"EarlyStopping starts after {self.start_epoch} epochs ...")

    def step(self, current_metrics: dict, epoch) -> bool:
        """
        在每个 epoch/step 结束后调用，用于更新各指标的最佳值以及判断是否提前停止。

        Args:
            current_metrics (dict): 当前 epoch/step 各指标的值，例如：
                {
                   "loss": 0.2345,
                   "NSPDK": 0.1984,
                   "FCD": 12.345,
                   "validity": 0.4321
                }

        Returns:
            stop (bool): 如果满足提前停止条件（根据 trigger_mode），返回 True；否则返回 False。
        """
        # 确保传入的指标都在 metrics_cfg 里
        assert set(self.metrics_cfg.keys()) <= set(current_metrics.keys())

        # 0) 先专门检测 loss=NaN 的情况，如果是就直接停止
        loss_val = current_metrics.get('loss', None)
        if loss_val is not None and (isinstance(loss_val, float) and math.isnan(loss_val)):
            if self.verbose:
                logger.info(f"[EarlyStopping] Detected loss=NaN at epoch {epoch}. Stopping immediately.")
            return True

        log_content = f"[Epoch {epoch}] "

        # 1) 遍历每个指标，判断是否“改善”
        for metric, cfg in self.metrics_cfg.items():
            mode = cfg['mode']
            min_delta = cfg['min_delta']
            current = current_metrics[metric]
            best = self.best_value[metric]

            # 判断“改善”条件
            improved = (current < best - min_delta) if mode == 'min' else (current > best + min_delta)
            if improved:
                self.best_value[metric] = current
                self.num_bad_epochs[metric] = 0
            elif epoch >= self.start_epoch:  # 只有在 start_epoch 之后才考虑 bad_epochs
                self.num_bad_epochs[metric] += 1
            log_content += f"{metric}: {current:.5f}(b={self.best_value[metric]:.5f},{self.num_bad_epochs[metric]}/{cfg['patience']}) |"


        if self.verbose:
            logger.info(log_content)

        # 2) 如果当前 epoch 小于 start_epoch，则不考虑提前停止
        if epoch < self.start_epoch:
            return False

        # 3) 根据 trigger_mode 判断是否要提前停止
        if self.trigger_mode == 'all':
            # 只有当所有指标都 bad_epochs >= patience，才停止
            for metric, cfg in self.metrics_cfg.items():
                if self.num_bad_epochs[metric] < cfg['patience']:
                    return False
            # 走到这里说明所有指标都 bad_epochs >= patience
            if self.verbose:
                logger.info("[EarlyStopping] Stopping because all metrics exceed patience.")
            return True
        else:
            # trigger_mode == 'any'
            # 只要任意一个指标的 bad_epochs >= patience，就停止
            for metric, cfg in self.metrics_cfg.items():
                if self.num_bad_epochs[metric] >= cfg['patience']:
                    if self.verbose:
                        logger.info(f"[EarlyStopping] Stopping because '{metric}' reaches patience = {cfg['patience']}")
                    return True
            return False



class CheckpointManager:
    """
    使用 Pareto 最优前沿维护 checkpoint，支持阈值过滤与最大保留数量。
    仅保留“不被任何其他 checkpoint 完全支配”的非劣解，当数量超过 max_size 时，
    使用“平均排名”策略做二次筛选。

    核心设计：
    1. _ALL_METRICS_INFO：类变量，列出了所有可能的指标及其“越大越好/越小越好”属性。
       - False 表示“该指标值越小越好”（如 loss、FCD、NSPDK）。
       - True  表示“该指标值越大越好”（如 validity）。
    2. __init__ 接受 metrics_to_monitor 集合，必须是 _ALL_METRICS_INFO.keys() 的子集，
       指定了当前会参与 Pareto 比较和阈值过滤的指标。未在此集合中的指标会被忽略。
    3. thresholds：可选参数，指定各监控指标的阈值，若某 checkpoint 的指标不满足阈值，即刻丢弃。
    4. max_size：Pareto 前沿最大保留数量，当超过该数量时，使用平均排名策略再做一次筛选。
    """

    # 所有可能的指标及其“越大越好/越小越好”属性(False 表示“值越小越好”)
    _ALL_METRICS_INFO = {
        'loss': False,
        'reconst_loss': False,
        'kld_loss': False,
        'NSPDK': False,
        'FCD': False,
        'validity': True
        # 如果将来需要新增指标，只需在这里扩展即可
    }

    def __init__(self, thresholds: dict = dict(), max_size: int = 10):
        """
        参数:

        - thresholds:         dict, 某些指标的阈值，如 {'NSPDK': 1.0, 'FCD': 5.0, 'validity': 0.8}，
                              若某 checkpoint 的任意指标不满足阈值，则直接丢弃。
                              thresholds.keys() 表示实际参与 Pareto 比较与阈值过滤的指标，必须是 _ALL_METRICS_INFO.keys() 的子集
        - max_size:           int, Pareto 前沿在任意时刻最大保留数量，
                              超过后以“平均排名”筛除剩余项。
        """
        # thresholds.keys() 表示实际参与 Pareto 比较与阈值过滤的指标，必须是 _ALL_METRICS_INFO.keys() 的子集
        if not thresholds.keys() <= self._ALL_METRICS_INFO.keys():
            raise ValueError(f"[CheckpointManager] metrics_to_monitor 必须是以下集合的子集: {set(self._ALL_METRICS_INFO.keys())}")

        # # 根据 metrics_to_monitor 从 ALL_METRICS_INFO 中提取需要监控的指标及属性
        self.max_size = max_size
        self.thresholds = thresholds
        self.metrics_info = {key: self._ALL_METRICS_INFO[key] for key in thresholds.keys()}


        # pareto_list 用来存放当前的 Pareto 非劣解，每个元素为 dict:
        #   {
        #     'metrics': {...},  # 该 checkpoint 的所有监控指标值，示例 {'NSPDK':0.4, 'FCD':3.2, 'validity':0.85}
        #     'path': str        # 该 checkpoint 在本地的完整存储路径，例如 "/.../model-epoch5.pt"
        #   }
        self.pareto_list = []

    def _meets_thresholds(self, metrics: dict) -> bool:
        """判断 metrics 是否满足 self.thresholds 中设定的阈值"""
        for key, thresh in self.thresholds.items():
            val = metrics.get(key)
            if val is None:
                # metrics 中不包含此 key，默认跳过
                continue
            rev = self.metrics_info[key] # (True表示越大越好，False 表示“值越小越好”)
            if (val < thresh and rev) or (val > thresh and not rev):
                return False

        return True

    def _is_dominated(self, m1: dict, m2: dict) -> bool:
        """
        返回 True，表示 m1 被 m2 完全支配；否则返回 False。
        m2 完全支配 m1 的条件是： m2 所有指标都不差于 m1， 且 m2 至少在一个指标优于 m1.


        """
        strictly_better = False

        for key, rev in self.metrics_info.items():
            v1, v2 = m1.get(key), m2.get(key)
            if v1 is None or v2 is None:
                continue # 如果某个指标在 m1 或 m2 中缺失，则跳过该指标判断

            if rev:
                # rev=true: 越大越好
                if v2 < v1:
                    return False
                if v2 > v1:
                    strictly_better = True
            else:
                # 越小越好
                if v2 > v1:
                    return False
                if v2 < v1:
                    strictly_better = True

        # 只有存在至少一个严格优的维度时，才算“完全支配”
        return strictly_better

    def _compute_average_ranks(self) -> dict:
        """
        对当前 self.pareto_list 中所有 entry，根据各指标计算排序并返回平均名次:
          返回 { entry_path: avg_rank, ... }
        排名规则:
          - reverse=True (越大越好): 数值越大，排名越靠前(rank=1)。
          - reverse=False(越小越好): 数值越小，排名越靠前(rank=1)。
        """
        n = len(self.pareto_list)
        if n == 0:
            return {}

        # 收集各指标的值
        metric_values = {key: [] for key in self.metrics_info}
        for entry in self.pareto_list:
            for key in self.metrics_info:
                metric_values[key].append(entry['metrics'].get(key, 0.0))

        ranks = {entry['path']: [] for entry in self.pareto_list}

        # 对每个指标分别排序并生成排名
        for key, rev in self.metrics_info.items():
            values = metric_values[key]
            if rev:
                sorted_indices = sorted(range(n), key=lambda i: values[i], reverse=True)
            else:
                sorted_indices = sorted(range(n), key=lambda i: values[i], reverse=False)

            key_ranks = [0] * n
            for rank, idx in enumerate(sorted_indices, start=1):
                key_ranks[idx] = rank

            for i, entry in enumerate(self.pareto_list):
                ranks[entry['path']].append(key_ranks[i])

        # 计算平均排名
        avg_ranks = {
            entry['path']: sum(ranks[entry['path']]) / len(ranks[entry['path']])
            for entry in self.pareto_list
        }
        return avg_ranks

    def update(self, ckpt_path: str, metrics: dict) -> (bool, set):
        """
        更新 Pareto 前沿。输入:
          - ckpt_path: 完整的 checkpoint 路径 (含目录 + 文件名)
          - metrics:   对应的指标字典, 必须包含 metrics_info 中指定的 key。

        流程:
          1. 阈值过滤: 若 metrics 中任何 key 不满足 self.thresholds，立即返回 (False, set())。
          2. Pareto 支配判断: 若新 metrics 被现有任一 entry 支配，则返回 (False, set())。
          3. 将新点插入 self.pareto_list.
          4. 移除被新点支配的旧 entry 并收集它们的路径到 removed_paths。
          5. 如果 self.pareto_list 长度 > max_size:
               a. 计算当前所有 entry 的平均排名 avg_ranks。
               b. 按 avg_rank 升序保留前 max_size 个，其余 entry 路径加入 removed_paths。
          6. keep_set = { entry['path'] for entry in self.pareto_list }
             should_save = ckpt_path in keep_set
             remove_set = removed_paths - keep_set
          返回: (should_save, remove_set)
        """
        # 1) 阈值过滤
        if not self._meets_thresholds(metrics):
            return False, set()

        # 2) 检查是否被 Pareto 集合中的某 entry 完全支配
        for entry in self.pareto_list:
            if self._is_dominated(metrics, entry['metrics']):
                # 若新加入的metric完全被现有 entry 支配，则不需要保存
                return False, set()

        # 3) 运行到这边说明需要加入新的metric, 然后移除被新点支配的旧 entry
        survivors = [{'metrics': metrics.copy(), 'path': ckpt_path}]
        removed_paths = []
        for entry in self.pareto_list:
            if self._is_dominated(entry['metrics'], metrics):
                removed_paths.append(entry['path'])
            else:
                survivors.append(entry)
        self.pareto_list = survivors


        # 4) 超出 max_size 后，用平均排名筛选
        if len(self.pareto_list) > self.max_size:
            avg_ranks = self._compute_average_ranks()
            # 按平均排名升序 (rank 越小越好)
            sorted_paths = sorted(avg_ranks.keys(), key=lambda p: avg_ranks[p])
            keep_paths = set(sorted_paths[: self.max_size])

            for entry in list(self.pareto_list):
                if entry['path'] not in keep_paths:
                    removed_paths.append(entry['path'])
            # 保留前 max_size 个 entry
            self.pareto_list = [e for e in self.pareto_list if e['path'] in keep_paths]

        # 5) 构造返回值
        keep_set = {entry['path'] for entry in self.pareto_list}
        should_save = (ckpt_path in keep_set)
        
        # remove_set 中不应该包含当前的 ckpt_path (因为它还没存盘, 删它会报错)
        remove_set = (set(removed_paths) - keep_set) - {ckpt_path}

        return should_save, remove_set
