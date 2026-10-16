# RingDiv preprocessing

`preprocess.py` cleans an assembled molecular CSV and exports a single `SMILES`
column. It removes salts and stereochemistry, keeps the largest fragment, filters
non-organic, charged and radical structures, and removes duplicates by InChI.

Deduplication uses InChI from the input structures and keeps the last occurrence.
Different input structures can therefore produce identical cleaned SMILES in the
output. Missing InChI values follow pandas' duplicate handling.

## Environment

Use a separate Python environment for preprocessing. The script has been tested
with Python 3.13.7, RDKit 2025.09.1 and pandas 2.2.3. Dependencies are listed in
`requirements.txt`.

## Usage

From the repository root, with the curation environment activated:

```bash
python -m pip install -r datasets/curation/requirements.txt
python datasets/curation/preprocess.py \
  --input /path/to/assembled_pool.csv \
  --output /path/to/new_directory/cleaned_smiles.csv
```

Input requires a `SMILES` column with no missing or non-string values. Unparsable
SMILES and structures that fail cleaning are dropped. The script processes the
CSV in memory and prints row counts and software versions when finished.

The output directory must already exist, and the output file must be new.
Retained rows keep their relative order, but row indices change when records are
removed. Generate new split indices for the output rather than reusing input
indices. Store data files outside this code directory.

