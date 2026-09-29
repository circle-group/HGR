# hgr.gvae — Grammar-VAE package.
#
# Intentionally empty: keeping __init__ lightweight lets utility submodules
# (loader, loss, data_loader) be imported without pulling in RDKit-backed
# decoder/runtime code. Import the full model classes explicitly when needed:
#   from hgr.gvae.model import GrammarSeq2SeqVAE
#   from hgr.gvae.latent2mol import Latent2MolDecoder
