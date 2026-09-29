# main_gvae_train.py

import os
import sys
import wandb
import torch

PRO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PRO_ROOT not in sys.path:
    sys.path.insert(0, PRO_ROOT)

import argparse
import logging
from hgr.utils.file_utils import PathManager
from hgr.utils.loader import set_env_from_config

set_env_from_config()
os.environ["_DEBUG_"] = "False"

logger = logging.getLogger(__name__)
logging.getLogger("hgr.grammar.smi").setLevel(logging.ERROR)


def _wandb_run_tag():
    """Return '_<run_id>' if wandb is active, else ''."""
    run_id = getattr(wandb.run, "id", None) if wandb.run else None
    return f"_{run_id}" if run_id else ""


def evaluate_prior_generation(config, gvae, device, train_smiles, test_smiles):
    from ringdiv import get_all_metrics
    from hgr.utils.eval_utils import decode_latents

    eval_cfg = config.eval
    bs = int(getattr(eval_cfg, "decode_batch_size", config.gvae.bs)) if eval_cfg else config.gvae.bs
    n_jobs = 16
    target_gen_num = 1024

    was_training = gvae.training
    gvae.eval()

    with torch.no_grad():
        z = gvae.sample_prior(target_gen_num, device=device)
        gen_mols, _ = decode_latents(gvae, z, chunk_size=bs, deterministic=True, sanitize_mols=True)

    if was_training:
        gvae.train()

    if not gen_mols:
        return {"Validity": 0.0, "Uniqueness": 0.0, "Novelty": 0.0, "FCD": float("inf")}

    rel = get_all_metrics(
        gen=gen_mols[:target_gen_num], test_smiles=test_smiles, train_smiles=train_smiles,
        metrics=['validity', 'novelty', 'uniqueness', 'fcd'], n_jobs=8,
        cache_dir=os.path.join(PathManager.DATA_DIR, "cache"), device=device,
    )
    return rel


def run_epoch(cfg, gvae, dataloader, loss_fn, device, epoch,
              optimizer=None, opt_fn=None, scheduler=None, is_train=False):
    from tqdm import tqdm

    if is_train:
        gvae.train()
    else:
        gvae.eval()

    total_loss, total_reconst, total_kld = 0.0, 0.0, 0.0
    dataset_size = len(dataloader.dataset)
    for batch_idx, batch in enumerate(
        tqdm(dataloader, total=len(dataloader), desc=f"{'Train' if is_train else 'Test'} epoch {epoch}",
             disable=not sys.stderr.isatty())
    ):
        in_batch, out_batch = batch[0].to(device), batch[1].to(device)
        global_step = epoch * len(dataloader) + batch_idx

        if is_train:
            mu, logvar = gvae.encode(in_batch)
            z = gvae.reparameterize(mu, logvar, stochastic=True)
            decoded = gvae.decode(z, out_batch)
            loss, reconst, kld = loss_fn(decoded, out_batch, mu, logvar, step=global_step)

            optimizer.zero_grad()
            loss.backward()
            opt_fn(optimizer, gvae.parameters(), global_step)
            optimizer.step()

            wandb.log(
                {"batch/loss": loss.item(), "batch/reconst": reconst.item(), "batch/kld": kld.item()},
                step=global_step,
            )
        else:
            with torch.no_grad():
                mu, logvar = gvae.encode(in_batch)
                z = gvae.reparameterize(mu, logvar, stochastic=True)
                decoded = gvae.decode(z, out_batch)
                loss, reconst, kld = loss_fn(decoded, out_batch, mu, logvar, step=global_step)

        total_loss += loss.item()
        total_reconst += reconst.item()
        total_kld += kld.item()
    if is_train and scheduler:
        scheduler.step()

    return {
        "loss": total_loss / dataset_size,
        "reconst": total_reconst / dataset_size,
        "kld": total_kld / dataset_size,
    }


def main(config):
    from hgr.utils.debug_utils import Timer
    from hgr.utils.file_utils import load_pickle, PathManager
    from hgr.utils.loader import (
        CheckpointManager,
        EarlyStopping,
        load_device,
        load_optimizer,
        load_seed,
        optimization_manager,
    )
    from hgr.gvae.data_loader import get_dataloaders
    from hgr.gvae.loss import GrammarVAELoss
    from hgr.gvae.model import GrammarSeq2SeqVAE
    from hgr.utils.data_splits import load_train_test_smiles

    load_seed(config.seed)
    device = load_device()
    tcfg = config.train

    with Timer("Loading grammar", enabled=True):
        grammar, rule_seq_list = load_pickle(os.path.join(PathManager.DATA_DIR, config.path.grammar_path))

    with Timer("Loading dataloader", enabled=True):
        train_loader, test_loader = get_dataloaders(config, rule_seq_list, grammar.num_prod_rule)

    with Timer("Loading train/test smiles", enabled=True):
        train_smiles, test_smiles = load_train_test_smiles(config)

    gvae = GrammarSeq2SeqVAE(hrg=grammar, cfg=config.gvae).to(device)

    optimizer, scheduler = load_optimizer(config, gvae.parameters())
    opt_fn = optimization_manager(config)
    loss_fn = GrammarVAELoss(
        prod_rule_corpus=grammar,
        anneal_steps=tcfg.anneal_steps,
        fix_steps=tcfg.fix_steps,
        beta=tcfg.beta,
    ).to(device)

    earlystop = EarlyStopping(
        {"loss": {"mode": "min"}, "reconst_loss": {"mode": "min"}, "kld_loss": {"mode": "min"}, "FCD": {"mode": "min"}},
        start_epoch=tcfg.earlystopping.start,
        default_patience=tcfg.earlystopping.patience,
        default_min_delta=0.01,
    )

    best_loss, best_reconst, best_kld = float("inf"), float("inf"), float("inf")
    best_fcd = float("inf")

    ckpt_manager = CheckpointManager({"loss": float("inf"), "reconst_loss": float("inf"), "kld_loss": float("inf"), "FCD": float("inf")}, max_size=4)

    # Plan C: selective prior (FCD) evaluation to save wall time on unpromising trials.
    last_prior_metrics = None   # cached fresh metrics, reused on skipped epochs for early-stop staleness
    prior_warmup_epoch = 5      # always skip before this (FCD is pure noise)
    prior_sparse_until = 10     # during [warmup, sparse_until): eval every `sparse_stride` epochs
    prior_sparse_stride = 2
    prior_promising_ratio = 1.2  # after sparse phase: eval only when test_loss ≤ prev_best × ratio
    prior_tail_epochs = 5       # force eval in the last N epochs for reliable early-stop signal

    for epoch in range(tcfg.n_iters):
        with Timer(f"Epoch {epoch}, Train time:"):
            train_metrics = run_epoch(config, gvae, train_loader, loss_fn, device, epoch, optimizer, opt_fn, scheduler, is_train=True)
        with Timer(f"Epoch {epoch}, Test time:"):
            test_metrics = run_epoch(config, gvae, test_loader, loss_fn, device, epoch, is_train=False)
        test_loss, test_reconst, test_kld = test_metrics["loss"], test_metrics["reconst"], test_metrics["kld"]
        prev_best_loss = best_loss  # snapshot BEFORE updating best_* for promising check
        best_loss, best_reconst, best_kld = min(best_loss, test_loss), min(best_reconst, test_reconst), min(best_kld, test_kld)

        # ── Plan C: decide whether to run prior evaluation this epoch ─────
        if epoch < prior_warmup_epoch:
            do_prior_eval = False                                   # too early
        elif epoch >= tcfg.n_iters - prior_tail_epochs:
            do_prior_eval = True                                    # force near end
        elif epoch < prior_sparse_until:
            do_prior_eval = (epoch % prior_sparse_stride == 0)      # sparse warm-up
        else:
            # Promising = current test_loss is within `ratio` of the best seen *before* this epoch
            do_prior_eval = test_loss <= prev_best_loss * prior_promising_ratio

        if do_prior_eval:
            with Timer(f"Epoch {epoch}, Prior eval time:", enabled=True):
                prior_metrics = evaluate_prior_generation(config, gvae, device, train_smiles, test_smiles)
            last_prior_metrics = prior_metrics
            fcd = float(prior_metrics.get("FCD", float("inf")))
            best_fcd = min(best_fcd, fcd)

            # ── Pareto checkpoint (only on fresh FCD) ────────────────────
            grammar_tag = str(getattr(config.grammar, "type", "")).upper() or "GVAE"
            ckpt_fname = (
                f"gvae-{grammar_tag}-recon{test_reconst:.3f}-kld{test_kld:.3f}"
                f"-tot{test_loss:.3f}-fcd{fcd:.4f}"
                f"-e{epoch}{_wandb_run_tag()} ({config.exp_time}).pth"
            )
            ckpt_path = os.path.join(PathManager.CKPT_DIR, ckpt_fname)
            should_save, to_remove = ckpt_manager.update(
                ckpt_path, {"loss": test_loss, "reconst_loss": test_reconst, "kld_loss": test_kld, "FCD": fcd},
            )
            if should_save:
                torch.save({"config": config, "gvae": gvae.state_dict()}, ckpt_path)
                logger.info("Saved checkpoint: %s", ckpt_path)
            for rem in to_remove:
                try:
                    os.remove(rem)
                except OSError as exc:
                    logger.error("Failed removing %s: %s", rem, exc)
        else:
            # No fresh FCD — skip checkpoint saving (stale FCD would corrupt Pareto ranking).
            # Pass current best_fcd to EarlyStopping so the FCD dimension keeps incrementing
            # bad_epochs (no improvement), letting trigger_mode='all' still fire on truly stalled trials.
            fcd = best_fcd

        if earlystop.step({"loss": test_loss, "reconst_loss": test_reconst, "kld_loss": test_kld, "FCD": fcd}, epoch=epoch):
            break

        # ── Logging ───────────────────────────────────────────────────────
        epoch_log = {"epoch": epoch}
        for prefix, metrics in [("train", train_metrics), ("test", test_metrics)]:
            for key, value in metrics.items():
                epoch_log[f"{prefix}/{key}"] = value
        epoch_log.update({
            "best/loss": best_loss, "best/reconst": best_reconst, "best/kld": best_kld,
            "best/FCD": best_fcd,
            "prior/evaluated": 1 if do_prior_eval else 0,
        })
        if do_prior_eval:
            epoch_log["prior/FCD"] = fcd
            for key in ("Validity", "Uniqueness", "Novelty"):
                epoch_log[f"prior/{key}"] = float(prior_metrics.get(key, 0.0))
        wandb.log(epoch_log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="configs/ringdiv300k/gvae_rsg.yaml",
        help="YAML path relative to the repository root (configs/...) or an absolute path",
    )
    args = parser.parse_args()

    from hgr.utils.debug_utils import make_wandb_init_config, update_wandb_config
    from hgr.utils.loader import init_exp, load_config

    config = load_config(args.config)
    sweep_id = os.getenv("WANDB_SWEEP_ID")
    if os.getenv("WANDB_SWEEP") or sweep_id:
        sweep_tag = sweep_id or "sweep"
        config.exp_name = f"{config.exp_name}_{sweep_tag}"
    env_cfg = init_exp(config)

    wandb_project = getattr(getattr(config, "wandb", None), "project", None) or "HGR-gvae"
    wandb.init(
        entity=env_cfg.wandb.entity,
        project=wandb_project,
        name=f"{os.getenv('WANDB_SWEEP', '')}{config.data.name}-{config.exp_time}",
        dir=PathManager.WANDB_DIR,
        config=make_wandb_init_config(config),
        allow_val_change=True,
    )
    config = update_wandb_config(config)

    try:
        main(config)
        print("Finished training.")
    except KeyboardInterrupt:
        print(">>> Caught SIGINT, finishing wandb...")
    finally:
        wandb.finish()
