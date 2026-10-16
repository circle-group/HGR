"""Historical RingDiv preprocessing extraction; see README for scope and caveats.

Input is an already assembled CSV with a SMILES column, NOT raw vendor catalogues.
InChI deduplication intentionally uses the original molecules, matching the source
notebook. This is not deduplication of the cleaned structures.
"""

import argparse
import json
import platform
from pathlib import Path


ORGANIC_ATOM_SET = {5, 6, 7, 8, 9, 15, 16, 17, 35, 53}


def keep_largest_fragment(mol):
    from rdkit import Chem

    fragments = Chem.GetMolFrags(mol, asMols=True)
    if not fragments:
        return None
    # max retains the first fragment on a size tie, as in the notebook loop.
    return Chem.MolToSmiles(max(fragments, key=lambda fragment: fragment.GetNumAtoms()))


def preprocess_mol(mol, remover):
    """Source cells 2–5: strip salts/stereo; retain neutral organic molecules."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    if mol is None:
        return None
    try:
        stripped = remover.StripMol(mol, dontRemoveEverything=True)
        smiles = Chem.MolToSmiles(stripped, isomericSmiles=False)
        if "." in smiles:
            smiles = keep_largest_fragment(Chem.MolFromSmiles(smiles))
        if not smiles:
            return None
        cleaned = Chem.MolFromSmiles(smiles)
        if cleaned is None:
            return None
        if not {atom.GetAtomicNum() for atom in cleaned.GetAtoms()}.issubset(ORGANIC_ATOM_SET):
            return None
        if any(atom.GetFormalCharge() != 0 for atom in cleaned.GetAtoms()):
            return None
        if Descriptors.NumRadicalElectrons(cleaned) > 0:
            return None
        return smiles
    except Exception:
        # The notebook treats molecule-processing failures as missing SMILES.
        return None


def safe_mol_to_inchi(mol):
    from rdkit import Chem

    try:
        return Chem.inchi.MolToInchi(mol) if mol is not None else None
    except Exception:
        return None


def preprocess_frame(frame):
    """Return the source notebook's SMILES-only export and stage row counts.

    Preserve order: parse -> clean SMILES -> original-molecule InChI -> keep
    last duplicate -> drop missing cleaned SMILES. InChI failures also follow
    pandas' original duplicate handling; they are counted, not silently fixed.
    """
    from rdkit.Chem import PandasTools
    from rdkit.Chem.SaltRemover import SaltRemover

    if "SMILES" not in frame.columns:
        raise ValueError("Input CSV must have a SMILES column")
    frame = frame[["SMILES"]].copy()
    if not frame["SMILES"].map(lambda value: isinstance(value, str)).all():
        raise ValueError("SMILES must contain strings only; missing/non-string entries are unsupported")
    counts = {"input_rows": len(frame)}
    PandasTools.AddMoleculeColumnToFrame(frame, "SMILES")
    frame = frame.dropna(subset=["ROMol"]).reset_index(drop=True)
    counts["parsed_rows"] = len(frame)
    remover = SaltRemover()
    frame["SMILES"] = frame["ROMol"].apply(lambda mol: preprocess_mol(mol, remover))
    counts["cleaning_failures"] = int(frame["SMILES"].isna().sum())
    frame["inchis"] = frame["ROMol"].apply(safe_mol_to_inchi)
    counts["missing_inchi_values"] = int(frame["inchis"].isna().sum())
    frame = frame.drop_duplicates(subset=["inchis"], keep="last")
    counts["after_inchi_deduplication"] = len(frame)
    frame = frame.dropna(subset=["SMILES"]).reset_index(drop=True)
    counts["output_rows"] = len(frame)
    return frame[["SMILES"]], counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Assembled CSV with SMILES")
    parser.add_argument("--output", type=Path, required=True, help="New SMILES-only CSV")
    args = parser.parse_args()
    # Validate paths before importing the scientific runtime or doing expensive work.
    if not args.input.is_file():
        parser.error(f"Input does not exist: {args.input}")
    if args.output.exists() or args.output.is_symlink():
        parser.error(f"Refusing to overwrite output: {args.output}")
    if not args.output.parent.is_dir():
        parser.error(f"Output directory does not exist: {args.output.parent}")

    import pandas as pd
    from rdkit import rdBase

    output, counts = preprocess_frame(pd.read_csv(args.input))
    # Exclusive creation also protects against an output appearing during processing.
    with args.output.open("x", newline="", encoding="utf-8") as handle:
        output.to_csv(handle, index=False)
    print(json.dumps({
        "workflow": "historical-preprocessing-only",
        "deduplication_basis": "original-molecule-inchi-keep-last",
        "versions": {"python": platform.python_version(), "rdkit": rdBase.rdkitVersion,
                     "pandas": pd.__version__},
        "counts": counts,
    }, indent=2))


if __name__ == "__main__":
    main()
