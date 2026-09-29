# Datasets

RingDiv and RingDiv300k will be publicly available upon publication. Obtain
third-party benchmarks from the original sources cited in the paper, following
their terms. Place inputs under `$ASSET_ROOT/datasets/` as follows.

| Dataset | Files under `datasets/` |
|---|---|
| RingDiv / RingDiv300k | `ringdiv/raw/ringdiv_property.csv`, `ringdiv/raw/ringdiv_test_idx.npz`; corresponding `ringdiv300k` files |
| QM9 | `qm9/raw/qm9_property.csv`, `qm9/raw/valid_idx_qm9.json` |
| ZINC250k | `zinc250k/raw/zinc250k_property.csv`, `zinc250k/raw/valid_idx_zinc250k.json` |
| MOSES | `moses/raw/moses_train.smiles`, `moses/raw/moses_test_scaffolds.smiles` |
| GuacaMol | `guacamol/raw/new_train.smiles`, `guacamol/raw/new_test.smiles` |
| BBBP / BACE / ClinTox / SIDER | `bbbp/raw/BBBP.csv`, `bace/raw/bace.csv`, `clintox/raw/clintox.csv`, `sider/raw/sider.csv` |
| Tox21 / ToxCast / HIV | `tox21/raw/tox21.csv`, `toxcast/raw/toxcast_data.csv`, `hiv/raw/HIV.csv` |
| ZINC2m | `zinc2m/raw/zinc2m_property.csv` (single `SMILES` column) |

Raw datasets are not included in Git. Original downloads may require the
preprocessing described in the paper; no automated download pipeline is provided.

Keep CSV row order and supplied split files unchanged. RingDiv NPZ `test_idx`
values are zero-based row positions excluding the header; training uses the
complement. QM9/ZINC250k `valid_idx` files specify the test split. MOSES uses
the scaffold test set. Data construction, sources, and preprocessing are described
in the paper.

## Rebuilding grammar artifacts

After installing HGR and placing the data, run from the repository root.
Set `ASSET_ROOT` and copy the bundled vocabulary inputs:

```bash
export ASSET_ROOT="$PWD"  # Or an absolute path to a separate asset directory.
export PYTHONHASHSEED=123
export WANDB_MODE=disabled
mkdir -p "$ASSET_ROOT/datasets/"{qm9,zinc250k,moses,guacamol,ringdiv300k}
cp configs/vocab/qm9_vocab_500.txt "$ASSET_ROOT/datasets/qm9/"
cp configs/vocab/zinc250k_vocab_500.txt "$ASSET_ROOT/datasets/zinc250k/"
for name in ringdiv300k moses guacamol; do
  cp configs/vocab/ringdiv300k_vocab_1000.txt "$ASSET_ROOT/datasets/$name/"
done
```

### Generation

```bash
python scripts/construct_grammar.py \
  --config configs/ringdiv300k/gvae_rsg.yaml --workers 1 \
  --output-config "$ASSET_ROOT/rebuilt-configs/ringdiv300k-gvae.yaml"
```

This checks molecular reconstruction and saves the grammar, partitions, and a
new YAML for `scripts/gvae_train.py --config`. Existing outputs are protected.
The default grammar filename is `RSG_ringdiv300k_rebuilt.pklz` in the dataset directory.
Use the corresponding GVAE config for QM9, ZINC250k, MOSES, or GuacaMol;
MIG additionally requires `--lifting_type MIG`. Increase `--workers` as needed.

For RingDiv, copy the RingDiv300k config, set `data.name: ringdiv`, and use an
absolute vocabulary path. For diffusion, set `path.grammar_path` in a copied
diffusion config to the new grammar and `path.gvae_ckpt_path` to a GVAE
checkpoint trained with it.

### Foundation models

```bash
python scripts/prepare_fm_data.py --config configs/pretrain.yaml \
  --workers 1 --output-config "$ASSET_ROOT/rebuilt-configs/zinc2m-fm.yaml"
python scripts/prepare_fm_data.py --config configs/pretrain_ringdiv.yaml \
  --workers 1 --output-config "$ASSET_ROOT/rebuilt-configs/ringdiv-fm.yaml"
python scripts/prepare_fm_data.py --config configs/pretrain_ringdiv_zinc2m.yaml \
  --base-config "$ASSET_ROOT/rebuilt-configs/ringdiv-fm.yaml" \
  --other-config "$ASSET_ROOT/rebuilt-configs/zinc2m-fm.yaml" \
  --output-config "$ASSET_ROOT/rebuilt-configs/joint-fm.yaml"
```

These commands require empty `processed/` directories and prepare graph features,
grammar, and rule tensors without training. The joint step also creates the
merged grammar and manifest in `datasets/ringdiv_zinc2m_rebuilt/`.
Pass the resulting YAML to `scripts/fm_pretrain.py --config`.

This workflow has passed small-sample checks only. Full-data reconstruction and
historical-checkpoint compatibility remain unverified; use rebuilt grammars for
training from scratch. Grammar artifacts and model checkpoints must be built or trained separately.
