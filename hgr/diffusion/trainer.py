# diffusion/trainer.py

import os
import torch
import wandb
import logging

from hgr.utils.file_utils import PathManager
from hgr.utils.ema import ExponentialMovingAverage
from hgr.utils.loader import load_device, load_optimizer, optimization_manager, load_seed, seed_context, EarlyStopping, CheckpointManager
from hgr.diffusion.data_loader import latent_dataloader
from hgr.diffusion.loader import load_scorenet
from hgr.diffusion.sampler import Sampler
from hgr.diffusion import losses
from hgr.diffusion.losses_mixture import MixtureMatching

logger = logging.getLogger(__name__)




class Trainer(object):
    def __init__(self, config, device=None):
        load_seed(config.seed)
        config.path.scorenet_ckpt_path = None  # 训练时清除旧的ckpt_path
        self.config = config
        self.device = load_device() if device is None else device

        # —— 数据加载器 ——
        # —— Prepare the DataLoader for latent fingerprints ——
        self.train_loader = latent_dataloader(config, self.device)
        # Ensure model embedding dim matches data dim
        assert config.scorenet.latent_dim == self.train_loader.dataset.tensors[0].shape[1], \
            f"[Error] Latent dimension mismatch: {config.scorenet.latent_dim} vs {self.train_loader.dataset.tensors[0].shape[1]}"

        self.score_net = load_scorenet(config.scorenet, self.device)
        self.optimizer, self.scheduler = load_optimizer(config, self.score_net.parameters())
        self.opt_fn  = optimization_manager(config)
        self.ema = ExponentialMovingAverage(self.score_net.parameters(), decay=config.train.ema_decay)
        # TODO: 后面检查一下将score_net直接传入loss_fn的初始化中是否合适
        sde_type = config.sde.type.lower()
        if sde_type in ['vpsde']:
            self.loss_fn = losses.DenoisingScoreMatching(config, is_train=True)
        elif sde_type in ['ou']:
            self.loss_fn = MixtureMatching(config, is_train=True)



        # If snapshot sampling is enabled, prepare sampler and metrics storage
        if config.train.snapshot_sampling:
            # NOTE 正式发布时这边需要删去
            # config.eval.batch_size = 512
            # config.eval.num_samples = 500
            config.eval.batch_size = 1024
            config.eval.num_samples = 1000
            self.sampler = Sampler(config, device=self.device, decoder_workers=1)
            if config.data.name == 'qm9':
                thresholds = {'NSPDK': 0.01, 'FCD': 3.0, 'validity': 0.95}
            elif config.data.name in ['zinc250k', 'ringdiv300k']:
                thresholds = {'NSPDK': 0.01, 'FCD': 4.0, 'validity': 0.95}
            elif config.data.name in ['moses', 'guacamol']:
                # These datasets currently do not report NSPDK in config.eval.metrics,
                # so keep snapshot checkpoints on FCD/validity only.
                thresholds = {'FCD': float('inf'), 'validity': 0.8}
            else:
                raise NotImplementedError

            self.snapshot_metrics = list(thresholds.keys())
            self.ckpt_manager = CheckpointManager(thresholds)

            # earlystop_metrics = {"loss": {"mode": "min"}}
            # for metric_name in self.snapshot_metrics:
            #     earlystop_metrics[metric_name] = {"mode": "max" if metric_name == "validity" else "min"}
            earlystop_metrics = {
                metric: {"mode": "max" if metric == "validity" else "min"}
                for metric in self.snapshot_metrics
            }
            self.earlystop = EarlyStopping(
                earlystop_metrics,
                start_epoch=config.train.earlystopping.start,
                default_patience=config.train.earlystopping.patience,
            )




    def train(self):
        global_step = 0
        min_loss = float('inf')
        best_snapshot_metrics = {}
        for metric_name in getattr(self, 'snapshot_metrics', []):
            best_snapshot_metrics[metric_name] = 0.0 if metric_name == 'validity' else float('inf')

        # -------- Training --------
        #for epoch in trange(self.config.train.num_epochs, desc='[Training epoch]', position=0, leave=False):
        for epoch in range(self.config.train.num_epochs):
            total_loss = 0.0
            for batch in self.train_loader:
                fingerprints = batch[0].to(self.device)
                global_step += 1
                self.optimizer.zero_grad()
                loss = self.loss_fn(self.score_net, fingerprints)
                loss.backward()
                self.opt_fn(self.optimizer, self.score_net.parameters(), global_step) # lr warmup + grad_clip
                self.optimizer.step()                                                 # 参数更新
                self.ema.update(self.score_net.parameters())                          # ema update
                total_loss += loss.item()

            # 学习率调度
            if self.scheduler:
                self.scheduler.step()

            is_best_loss = total_loss < min_loss
            if is_best_loss:
                min_loss = total_loss
            wandb.log({'loss': total_loss, 'best/min_loss': min_loss}, step=epoch)
            # logger.info(f"Epoch {epoch} | Loss: {total_loss:.4f} (min {min_loss:.4f})")

            if epoch >= self.config.train.snapshot_freq and (is_best_loss or epoch % self.config.train.snapshot_freq == 0):
                # 照理说应该每隔 snapshot_freq保存一次模型，但是为了节省空间暂时没有这么做
                # ckpt_path = os.path.join(PathManager.CKPT_DIR, f'ScoreNet-{self.config.exp_time}-{total_loss}-{epoch}.pt')
                # torch.save({'config': self.config, 'scorenet': self.score_net.state_dict(), 'ema': self.ema.state_dict()},ckpt_path)

                # ---------------------------- snapshot sampling  ----------------------------
                if self.config.train.snapshot_sampling:
                    with torch.no_grad(), seed_context(self.config.seed):
                        self.ema.store(self.score_net.parameters())
                        try:
                            self.ema.copy_to(self.score_net.parameters())
                            gen_metrics = self.sampler.sample(score_net=self.score_net)
                            if 'Validity' in gen_metrics and 'validity' not in gen_metrics:
                                gen_metrics['validity'] = gen_metrics['Validity']
                            for metric_name in list(gen_metrics):
                                if metric_name.startswith('VUN'):
                                    del gen_metrics[metric_name]
                        finally:
                            self.ema.restore(self.score_net.parameters())

                    for metric_name in self.snapshot_metrics:
                        if metric_name not in gen_metrics:
                            continue
                        if metric_name == 'validity':
                            if gen_metrics[metric_name] > best_snapshot_metrics[metric_name]:
                                best_snapshot_metrics[metric_name] = gen_metrics[metric_name]
                            gen_metrics['best/max_validity'] = best_snapshot_metrics[metric_name]
                        else:
                            if gen_metrics[metric_name] < best_snapshot_metrics[metric_name]:
                                best_snapshot_metrics[metric_name] = gen_metrics[metric_name]
                            gen_metrics[f'best/min_{metric_name}'] = best_snapshot_metrics[metric_name]
                    wandb.log(gen_metrics, step=epoch)

                    # ✅ save and remove checkpoints
                    if 'NSPDK' in gen_metrics:
                        fname = f"ScoreNet-ns{gen_metrics['NSPDK']:.5f}-fcd{gen_metrics['FCD']:.4f}-val{gen_metrics['validity']:.4f}({self.config.exp_time}-{epoch}).pt"
                    else:
                        fname = f"ScoreNet-fcd{gen_metrics['FCD']:.4f}-val{gen_metrics['validity']:.4f}({self.config.exp_time}-{epoch}).pt"
                    ckpt_path = os.path.join(PathManager.CKPT_DIR, fname)
                    should_save, to_remove = self.ckpt_manager.update(ckpt_path, gen_metrics)
                    if should_save:
                        try:
                            torch.save({
                                'config': self.config,
                                'scorenet': self.score_net.state_dict(),
                                'ema': self.ema.state_dict()
                            }, ckpt_path)
                            logger.info(f"Saved checkpoint: {ckpt_path}")
                        except Exception as e:
                            logger.error(f"Failed to save checkpoint: {e}")
                            pass

                    for rem in to_remove:
                        try:
                            os.remove(rem)
                            logger.info(f"Removed old checkpoint: {rem}")
                        except OSError as e:
                            logger.error(f"Failed removing {rem}: {e}")
                            pass


                    # ✅ Early stopping (print epoch metrics within the early stopping check)
                    gen_metrics.update({'loss': total_loss})
                    if self.earlystop.step(gen_metrics, epoch=epoch):
                        break

                    # logger.info(f"Epoch {epoch} | NSPDK: {gen_rel['NSPDK']:.5f} (min {min_NSPDK:.5f}) | "
                    #              f"FCD: {gen_rel['FCD']:.4f} (min {min_FCD:.4f}) | "
                    #              f"Valid: {gen_rel['validity']:.4f} (max {max_validity:.4f})")
