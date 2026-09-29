# latent2mol.py  —  Grammar VAE latent → molecule decoder wrapper

import torch
from multiprocessing import get_context
from concurrent.futures import ProcessPoolExecutor, as_completed

from hgr.utils.file_utils import load_pickle
from hgr.utils.debug_utils import suppress_stderr
from hgr.grammar.smi import hg_to_mol
from hgr.gvae.model import GrammarSeq2SeqVAE
from hgr.gvae.loader import load_gvae_ckpt_cfg

global _GVAE_MODEL_, _DEVICE_
_worker_initialized = False


def chunk_tensor(x: torch.Tensor, chunk_size: int):
    for i in range(0, x.size(0), chunk_size):
        yield x[i : i + chunk_size]


def _init_worker(grammar, gvae_state, gvae_cfg, device_str):
    global _worker_initialized, _GVAE_MODEL_, _DEVICE_
    if _worker_initialized:
        return
    _worker_initialized = True
    _DEVICE_ = torch.device(device_str)
    grammar.to(_DEVICE_)
    _GVAE_MODEL_ = GrammarSeq2SeqVAE(hrg=grammar, cfg=gvae_cfg).to(_DEVICE_)
    _GVAE_MODEL_.load_state_dict(gvae_state)
    _GVAE_MODEL_.eval()


def _decode_chunk(args):
    latent_chunk, num_pad, deterministic = args
    z = latent_chunk.to(_DEVICE_)
    with torch.no_grad():
        finished, hg_list = _GVAE_MODEL_.decode(z, deterministic=deterministic, return_hg_list=True)
        if num_pad:
            finished = finished[:-num_pad]
            hg_list = hg_list[:-num_pad]

    gen_mols = []
    for tag, hg in zip(finished, hg_list):
        if not tag:
            continue
        try:
            with suppress_stderr():
                mol = hg_to_mol(hg)
                gen_mols.append(mol)
        except Exception:
            gen_mols.append(None)
            continue
    return gen_mols


class Latent2MolDecoder:
    def __init__(self, grammar_path, gvae_ckpt_path, device_str='cpu', workers=1):
        self.workers = workers
        grammar = load_pickle(grammar_path)
        if isinstance(grammar, tuple):
            grammar = grammar[0]
        gvae_state, self.gvae_cfg = load_gvae_ckpt_cfg(
            gvae_ckpt_path, device=torch.device(device_str), expected_type="GVAE",
        )
        self.executor = ProcessPoolExecutor(
            mp_context=get_context("spawn"),
            initializer=_init_worker,
            initargs=(grammar, gvae_state['gvae'], self.gvae_cfg, device_str),
            max_workers=workers,
        )
        if workers <= 1:
            _init_worker(grammar, gvae_state['gvae'], self.gvae_cfg, device_str)

    def decode(self, latent_tensor, preserve_order=True):
        batch_size = self.gvae_cfg.bs
        chunks = []
        for sub_z in chunk_tensor(latent_tensor, batch_size):
            num_pad = batch_size - sub_z.size(0)
            if num_pad > 0:
                sub_z = torch.cat([sub_z, sub_z.new_zeros((num_pad,) + sub_z.shape[1:])], dim=0)
            chunks.append((sub_z, num_pad, True))

        gen_mols = []
        if self.workers <= 1:
            for args in chunks:
                gen_mols.extend(_decode_chunk(args))
            return gen_mols

        if preserve_order:
            for mols in self.executor.map(_decode_chunk, chunks):
                gen_mols.extend(mols)
        else:
            futures = [self.executor.submit(_decode_chunk, c) for c in chunks]
            for fut in as_completed(futures):
                gen_mols.extend(fut.result())
        return gen_mols

    __call__ = decode

    def close(self):
        self.executor.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
