from rdkit import Chem

# ----------------------------- Constants -----------------------------
NUM_ATOM_TYPE = 120       # including the extra mask tokens
NUM_CHIRALITY_TAG = 8     # +5

NUM_BOND_TYPE = 24        # including aromatic and self-loop edge, and extra masked tokens  (+18)
NUM_BOND_DIRECTION = 7    # +4

# Grammar 构造时使用
PAD, BOS, EOS, MASK, OFFSET = 0, 1, 2, 3, 4


# Mapping from dataset name to number of tasks (unchanged)
DATASET_NUM_TASKS = {
    # classification
    'tox21': 12,
    'hiv': 1,
    'pcba': 128,
    'muv': 17,
    'bace': 1,
    'bbbp': 1,
    'toxcast': 617,
    'sider': 27,
    'clintox': 2,
    'mutag': 1,
    # regression
    'esol': 1,
    'freesolv': 1,
    'lipophilicity': 1,
}

# allowable node and edge features
ALLOW_FEATURES = {
    'atomic_num_list' : list(range(1, 119)),
    'formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'chirality_list' : [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER,        
        Chem.rdchem.ChiralType.CHI_ALLENE,
        Chem.rdchem.ChiralType.CHI_OCTAHEDRAL,
        Chem.rdchem.ChiralType.CHI_SQUAREPLANAR,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL,
        Chem.rdchem.ChiralType.CHI_TRIGONALBIPYRAMIDAL,
    ],
    'hybridization_list' : [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'implicit_valence_list' : [0, 1, 2, 3, 4, 5, 6],
    'degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'bonds' : [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,        
        Chem.rdchem.BondType.UNSPECIFIED,
        Chem.rdchem.BondType.QUADRUPLE,
        Chem.rdchem.BondType.QUINTUPLE,
        Chem.rdchem.BondType.HEXTUPLE,
        Chem.rdchem.BondType.ONEANDAHALF,
        Chem.rdchem.BondType.TWOANDAHALF,
        Chem.rdchem.BondType.THREEANDAHALF,
        Chem.rdchem.BondType.FOURANDAHALF,
        Chem.rdchem.BondType.FIVEANDAHALF,
        Chem.rdchem.BondType.IONIC,
        Chem.rdchem.BondType.HYDROGEN,
        Chem.rdchem.BondType.THREECENTER,
        Chem.rdchem.BondType.DATIVEONE,
        Chem.rdchem.BondType.DATIVE,
        Chem.rdchem.BondType.DATIVEL,
        Chem.rdchem.BondType.DATIVER,
        Chem.rdchem.BondType.OTHER,
        Chem.rdchem.BondType.ZERO
    ],
    'bond_dirs' : [ # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT,        
        Chem.rdchem.BondDir.BEGINDASH,
        Chem.rdchem.BondDir.BEGINWEDGE,
        Chem.rdchem.BondDir.EITHERDOUBLE,
        Chem.rdchem.BondDir.UNKNOWN
        
    ]
}