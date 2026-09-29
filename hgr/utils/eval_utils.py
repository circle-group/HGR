"""Shared utilities for GVAE evaluation scripts."""

from __future__ import annotations

import csv
import os
import re
from typing import List

import torch

from hgr.utils.file_utils import ensure_parent_dir, sanitize_filename

CKPT_RE = re.compile(
    r"recon(?P<recon>-?\d+(?:\.\d+)?)"
    r"-kld(?P<kld>-?\d+(?:\.\d+)?)"
    r"-tot(?P<tot>-?\d+(?:\.\d+)?)"
    r"(?:-e(?P<epoch>\d+))? \((?P<stamp>[^)]+)\)"
)

PREFERRED_METRICS = [
    "Validity",
    "Uniqueness",
    "Novelty",
    "VUN10000",
    "FCD",
    "NSPDK",
    "K_OR",
    "K_BF",
    "K_FR",
    "K_R",
    "KLdiv",
    "MW",
    "logP",
    "SA",
    "QED",
    "IntDiv",
    "Filters",
    "Frag/Test",
    "Scaf/Test",
]

BASE_EVAL_FIELDS = [
    "mode",
    "sigma",
    "interp_lambda",
    "interp_alpha",
    "status",
    "ckpt_name",
    "recon",
    "kld",
    "tot",
    "epoch",
    "stamp",
    "split",
    "limit",
    "seed",
    "attempts",
    "finished",
    "finished_rate",
    "gen_count",
    "attempt_validity",
    "nn_dist_mean",
    "nn_dist_median",
    "encode_seconds",
    "decode_seconds",
    "eval_seconds",
    "out_smi",
    "ckpt",
    "error",
]


def safe_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def chunk_tensor(x: torch.Tensor, chunk_size: int):
    for start in range(0, x.size(0), chunk_size):
        yield x[start : start + chunk_size]


def decode_latents(
    model,
    latent_tensor: torch.Tensor,
    chunk_size: int,
    deterministic: bool = True,
    sanitize_mols: bool = False,
):
    """Decode latent vectors into molecules via the GVAE decoder.

    Works with both V1 (fixed batch_size) and V4 (flexible batch_size) models.
    For V1, chunks are padded up to chunk_size; for V4, no padding is needed.
    """
    from hgr.grammar.smi import hg_to_mol
    from hgr.utils.debug_utils import suppress_stderr
    Chem = None
    if sanitize_mols:
        from rdkit import Chem

    gen_mols = []
    valid_attempts = 0
    finished_attempts = 0
    conversion_fail = 0
    total_attempts = latent_tensor.size(0)
    needs_padding = hasattr(model, "batch_size")  # V1 requires fixed batch size

    for sub_z in chunk_tensor(latent_tensor, chunk_size):
        num_pad = 0
        if needs_padding:
            num_pad = chunk_size - sub_z.size(0)
            if num_pad > 0:
                sub_z = torch.cat([sub_z, sub_z.new_zeros(num_pad, *sub_z.shape[1:])], dim=0)

        with torch.no_grad():
            finished, hg_list = model.decode(sub_z, deterministic=deterministic, return_hg_list=True)

        if num_pad > 0:
            finished = finished[:-num_pad]
            hg_list = hg_list[:-num_pad]

        finished_attempts += int(finished.sum().item())
        for tag, hg in zip(finished.tolist(), hg_list):
            if not tag:
                continue
            try:
                with suppress_stderr():
                    mol = hg_to_mol(hg)
                if mol is None:
                    raise ValueError("hg_to_mol returned None")
                if sanitize_mols:
                    Chem.SanitizeMol(mol)
                gen_mols.append(mol)
                valid_attempts += 1
            except Exception:
                conversion_fail += 1
                continue

    stats = {
        "attempts": total_attempts,
        "finished": finished_attempts,
        "unfinished": total_attempts - finished_attempts,
        "valid_count": valid_attempts,
        "conversion_fail": conversion_fail,
        "attempt_validity": valid_attempts / total_attempts if total_attempts > 0 else 0.0,
        "finished_rate": finished_attempts / total_attempts if total_attempts > 0 else 0.0,
    }
    return gen_mols, stats


def select_metric_cols(metric_keys: set) -> List[str]:
    preferred = [name for name in PREFERRED_METRICS if name in metric_keys]
    return preferred + sorted(metric_keys - set(preferred))


def write_summary_row(summary_out: str | None, row: dict, extra_base_fields: List[str] | None = None):
    if summary_out is None:
        return
    base = list(BASE_EVAL_FIELDS)
    if extra_base_fields:
        insert_pos = base.index("status") if "status" in base else 0
        for field in reversed(extra_base_fields):
            if field not in base:
                base.insert(insert_pos, field)
    metric_fields = [f for f in PREFERRED_METRICS if f in row]
    metric_fields += sorted(f for f in row if f not in set(base) and f not in set(metric_fields))
    fieldnames = base + metric_fields

    with open(summary_out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({f: row.get(f, "") for f in fieldnames})


def write_markdown_table(rows: list, metric_cols: list, out_path: str, title: str = "Evaluation Summary"):
    md_cols = [
        "mode", "status", "ckpt_name", "recon", "kld", "tot", "epoch",
    ] + [c for c in ["Validity", "VUN10000", "FCD", "NSPDK", "K_OR", "K_BF", "K_FR", "K_R", "KLdiv"]
         if c in metric_cols]

    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(f"# {title}\n\n")
        handle.write(f"Rows: {len(rows)}\n\n")
        handle.write("| " + " | ".join(md_cols) + " |\n")
        handle.write("|" + "|".join(["---"] * len(md_cols)) + "|\n")
        for row in rows:
            values = []
            for col in md_cols:
                value = row.get(col, "")
                if isinstance(value, float):
                    values.append(f"{value:.6f}")
                else:
                    values.append(str(value))
            handle.write("| " + " | ".join(values) + " |\n")


def build_output_paths(args, config, ckpt_path: str, eval_subdir: str, mode_tag: str | None = None):
    """Build save_path and summary_out from args, config, and checkpoint path.

    Parameters
    ----------
    eval_subdir : str
        Subdirectory under results/eval/ (e.g. "gvae_posterior", "gvae_ablation").
    mode_tag : str, optional
        Override the auto-derived mode tag used in file/directory naming.
    """
    from hgr.utils.file_utils import PathManager

    if mode_tag is None:
        if args.mode == "noisy_mu":
            mode_tag = f"noisy_mu_s{args.sigma:g}"
        elif args.mode == "interp_mu_prior":
            mode_tag = f"interp_mu_prior_l{args.interp_lambda:g}"
        else:
            mode_tag = args.mode

    if args.out_dir is None:
        from datetime import datetime
        ckpt_name = sanitize_filename(os.path.basename(ckpt_path).replace(".pth", ""))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_filename = f"{ckpt_name}_{mode_tag}_n{args.limit}_seed{config.seed}({timestamp}).smi"
        save_parts = [PathManager.RESULTS_ROOT, "eval", eval_subdir, config.data.name]
        if getattr(config, "exp_name", None):
            save_parts.append(config.exp_name)
        save_parts.append(mode_tag)
        save_dir = os.path.join(*save_parts)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, save_filename)
    else:
        save_path = args.out_dir
        ensure_parent_dir(save_path)

    summary_out = getattr(args, "summary_out", None)
    if summary_out is not None:
        ensure_parent_dir(summary_out)

    return save_path, summary_out


def parse_ckpt_metadata(ckpt_name: str) -> dict:
    match = CKPT_RE.search(ckpt_name)
    if match:
        return match.groupdict()
    return {"recon": "", "kld": "", "tot": "", "epoch": "", "stamp": ""}
