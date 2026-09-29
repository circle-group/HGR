
import os

import torch
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data
from hgr.foundation.data_utils.mol_defs import ALLOW_FEATURES
from hgr.utils.mol_utils import remove_salt_stereo


# ---- Fast lookup maps (build once) ----
ATOM_NUM_TO_IDX   = {z:i for i, z in enumerate(ALLOW_FEATURES['atomic_num_list'])}
CHIRAL_TO_IDX     = {c:i for i, c in enumerate(ALLOW_FEATURES['chirality_list'])}
BOND_TYPE_TO_IDX  = {b:i for i, b in enumerate(ALLOW_FEATURES['bonds'])}
BOND_DIR_TO_IDX   = {d:i for i, d in enumerate(ALLOW_FEATURES['bond_dirs'])}


def mol_to_graph_data_obj_simple(mol, numThreads=1):
    """
    Convert an RDKit Mol to a PyG Data with:
      - x: [num_atoms, 2]  (atomic_num_idx, chirality_idx)
      - edge_index, edge_attr (bond_type_idx, bond_dir_idx)
      - descriptors, descriptors_ss, maccs
      - esomeprazole: RDKit Mol with H for conformer embedding
      - min1pos, min2pos, min3pos: torch.FloatTensor [num_atoms, 3]
    """
    try:
        # smiles = Chem.MolToSmiles(mol, canonical=True)
        smiles = remove_salt_stereo(mol)
        mol = Chem.MolFromSmiles(smiles)
        mol_prop = Chem.RemoveHs(mol, implicitOnly=True)
        try:      
            Chem.SanitizeMol(mol_prop)  # 完整清洗：会建立芳香性、价态、环信息等
        except Exception:
            # 若完整清洗失败，至少保证环信息和属性缓存可用
            Chem.rdmolops.FastFindRings(mol_prop)          # 初始化 RingInfo
            mol_prop.UpdatePropertyCache(strict=False)      # 补属性缓存
            # 可选：尽量 kekulize，失败就算了
            try:
                Chem.Kekulize(mol_prop, clearAromaticFlags=True)
            except Exception:
                pass

        
        # ----- STEP 1. atoms (num_atom_features = 2: atom type,  chirality tag) ----- #
        atom_features_list = []  
        # 本地变量绑定加速属性访问
        atom_map = ATOM_NUM_TO_IDX
        ch_map   = CHIRAL_TO_IDX
        for atom in mol_prop.GetAtoms():
            atom_feature = [atom_map[atom.GetAtomicNum()], ch_map[atom.GetChiralTag()]]
            atom_features_list.append(atom_feature)
        x = torch.tensor(np.array(atom_features_list), dtype=torch.long)
        if x.shape[0] < 3: ## invaild feature remove (addition)
            return None, None,"Invalid_atom_feature"

        # ----- STEP2. bonds ----- #
        num_bond_features = 2   # bond type, bond direction
        if len(mol_prop.GetBonds()) > 0: # mol has bonds
            edges_list = []
            edge_features_list = []
            bt_map, bd_map = BOND_TYPE_TO_IDX, BOND_DIR_TO_IDX
            for bond in mol_prop.GetBonds():
                i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                edges_list += [(i, j), (j, i)]
                edge_feature = [bt_map[bond.GetBondType()], bd_map[bond.GetBondDir()]]
                edge_features_list += [edge_feature, edge_feature]

            # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
            edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

            # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
            edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)
        else:   # mol has no bonds
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        return data, smiles, None

    except Exception:
        return None, None, "Unknown error in mol2PyG_conf"


def process_bbbp(input_path):
    smiles_list, labels = _load_bbbp_dataset(input_path)

    data_list, data_smiles_list = [], []
    for i in range(len(smiles_list)):
        rdkit_mol = AllChem.MolFromSmiles(smiles_list[i])
        if rdkit_mol != None:
            # # convert aromatic bonds to double bonds
            # Chem.SanitizeMol(rdkit_mol,
            #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            data, smiles, err = mol_to_graph_data_obj_simple(rdkit_mol)
            if data == None: 
                print("{} is removed".format(smiles_list[i]))
                continue ## addition
            # manually add mol id
            data.id = torch.tensor(
                [i])  # id here is the index of the mol in
            # the dataset
            data.y = torch.tensor([labels[i]], dtype=torch.long)
            data_list.append(data)
            data_smiles_list.append(smiles)

    return data_list, data_smiles_list

def _load_bbbp_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    # rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]

    # preprocessed_rdkit_mol_objs_list = [m if m != None else None for m in
    #                                                       rdkit_mol_objs_list]
    # preprocessed_smiles_list = [AllChem.MolToSmiles(m) if m != None else
    #                             None for m in preprocessed_rdkit_mol_objs_list]
    labels = input_df['p_np']
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # there are no nans
    # assert len(smiles_list) == len(preprocessed_rdkit_mol_objs_list)
    # assert len(smiles_list) == len(preprocessed_smiles_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, labels.values


def process_tox21(input_path):
    smiles_list, labels = _load_tox21_dataset(input_path)

    data_list, data_smiles_list = [], []
    for i in range(len(smiles_list)):
        rdkit_mol = AllChem.MolFromSmiles(smiles_list[i])
        ## convert aromatic bonds to double bonds
        #Chem.SanitizeMol(rdkit_mol,
                            #sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        data, smiles, err = mol_to_graph_data_obj_simple(rdkit_mol)
        if data == None: 
            print("{} is removed".format(smiles_list[i]))
            continue ## addition
        # manually add mol id
        data.id = torch.tensor(
            [i])  # id here is the index of the mol in
        # the dataset
        data.y = torch.tensor(labels[i, :])
        data_list.append(data)
        data_smiles_list.append(smiles)
    
    return data_list, data_smiles_list

def _load_tox21_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    # rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    tasks = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
       'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
    labels = input_df[tasks]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # convert nan to 0
    labels = labels.fillna(0)
    assert len(smiles_list) == len(labels)
    return smiles_list, labels.values


def process_toxcast(input_path):    
    smiles_list, rdkit_mol_objs, labels =_load_toxcast_dataset(input_path)
    
    data_list, data_smiles_list = [], []
    for i in range(len(smiles_list)):
        rdkit_mol = rdkit_mol_objs[i]
        if rdkit_mol != None:
            # # convert aromatic bonds to double bonds
            # Chem.SanitizeMol(rdkit_mol,
            #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            data, smiles, err = mol_to_graph_data_obj_simple(rdkit_mol)
            if data == None: 
                print("{} is removed".format(smiles_list[i]))
                continue ## addition
            # manually add mol id
            data.id = torch.tensor(
                [i])  # id here is the index of the mol in
            # the dataset
            data.y = torch.tensor(labels[i, :])
            data_list.append(data)
            data_smiles_list.append(smiles)
    
    return data_list, data_smiles_list

def _load_toxcast_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    # NB: some examples have multiple species, some example smiles are invalid
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    # Some smiles could not be successfully converted
    # to rdkit mol object so them to None
    preprocessed_rdkit_mol_objs_list = [m if m != None else None for m in
                                        rdkit_mol_objs_list]
    preprocessed_smiles_list = [AllChem.MolToSmiles(m) if m != None else
                                None for m in preprocessed_rdkit_mol_objs_list]
    tasks = list(input_df.columns)[1:]
    labels = input_df[tasks]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # convert nan to 0
    labels = labels.fillna(0)
    assert len(smiles_list) == len(preprocessed_rdkit_mol_objs_list)
    assert len(smiles_list) == len(preprocessed_smiles_list)
    assert len(smiles_list) == len(labels)
    return preprocessed_smiles_list, preprocessed_rdkit_mol_objs_list, \
           labels.values


def process_sider(input_path):
    smiles_list, rdkit_mol_objs, labels =_load_sider_dataset(input_path)
    
    data_list, data_smiles_list = [], []
    for i in range(len(smiles_list)):
        rdkit_mol = rdkit_mol_objs[i]
        # # convert aromatic bonds to double bonds
        # Chem.SanitizeMol(rdkit_mol,
        #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        data, smiles, err = mol_to_graph_data_obj_simple(rdkit_mol)
        if data == None: 
            print("{} is removed".format(smiles_list[i]))
            continue ## addition
        # manually add mol id
        data.id = torch.tensor(
            [i])  # id here is the index of the mol in
        # the dataset
        data.y = torch.tensor(labels[i, :])
        data_list.append(data)
        data_smiles_list.append(smiles)
    
    return data_list, data_smiles_list

def _load_sider_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    tasks = ['Hepatobiliary disorders',
       'Metabolism and nutrition disorders', 'Product issues', 'Eye disorders',
       'Investigations', 'Musculoskeletal and connective tissue disorders',
       'Gastrointestinal disorders', 'Social circumstances',
       'Immune system disorders', 'Reproductive system and breast disorders',
       'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
       'General disorders and administration site conditions',
       'Endocrine disorders', 'Surgical and medical procedures',
       'Vascular disorders', 'Blood and lymphatic system disorders',
       'Skin and subcutaneous tissue disorders',
       'Congenital, familial and genetic disorders',
       'Infections and infestations',
       'Respiratory, thoracic and mediastinal disorders',
       'Psychiatric disorders', 'Renal and urinary disorders',
       'Pregnancy, puerperium and perinatal conditions',
       'Ear and labyrinth disorders', 'Cardiac disorders',
       'Nervous system disorders',
       'Injury, poisoning and procedural complications']
    labels = input_df[tasks]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values

def process_clintox(input_path):
    smiles_list, rdkit_mol_objs, labels =_load_clintox_dataset(input_path)

    data_list, data_smiles_list = [], []
    for i in range(len(smiles_list)):
        rdkit_mol = rdkit_mol_objs[i]
        if rdkit_mol != None:
            # # convert aromatic bonds to double bonds
            # Chem.SanitizeMol(rdkit_mol,
            #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            data, smiles, err = mol_to_graph_data_obj_simple(rdkit_mol)
            if data == None: 
                print("{} is removed".format(smiles))
                continue ## addition
            # manually add mol id
            data.id = torch.tensor(
                [i])  # id here is the index of the mol in
            # the dataset
            data.y = torch.tensor(labels[i, :])
            data_list.append(data)
            data_smiles_list.append(smiles)

    return data_list, data_smiles_list

def _load_clintox_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]

    preprocessed_rdkit_mol_objs_list = [m if m != None else None for m in
                                        rdkit_mol_objs_list]
    preprocessed_smiles_list = [AllChem.MolToSmiles(m) if m != None else
                                None for m in preprocessed_rdkit_mol_objs_list]
    tasks = ['FDA_APPROVED', 'CT_TOX']
    labels = input_df[tasks]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # there are no nans
    assert len(smiles_list) == len(preprocessed_rdkit_mol_objs_list)
    assert len(smiles_list) == len(preprocessed_smiles_list)
    assert len(smiles_list) == len(labels)
    return preprocessed_smiles_list, preprocessed_rdkit_mol_objs_list, \
           labels.values
# input_path = 'dataset_conf/clintox/raw/clintox.csv'
# smiles_list, rdkit_mol_objs_list, labels = _load_clintox_dataset(input_path)

def process_hiv(input_path):
    smiles_list, labels =_load_hiv_dataset(input_path)
    data_list, data_smiles_list = [], []
    for i in range(len(smiles_list)):
        rdkit_mol = AllChem.MolFromSmiles(smiles_list[i])
        # # convert aromatic bonds to double bonds
        # Chem.SanitizeMol(rdkit_mol,
        #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        data, smiles, err = mol_to_graph_data_obj_simple(rdkit_mol)
        if data == None: 
            print("{} is removed".format(smiles_list[i]))
            continue ## addition
        # manually add mol id
        data.id = torch.tensor(
            [i])  # id here is the index of the mol in
        # the dataset
        data.y = torch.tensor([labels[i]])
        data_list.append(data)
        data_smiles_list.append(smiles)
    
    return data_list, data_smiles_list

def _load_hiv_dataset(input_path):
    """
    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    labels = input_df['HIV_active']
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # there are no nans
    assert len(smiles_list) == len(labels)
    return smiles_list, labels.values


def process_bace(input_path):
    smiles_list, rdkit_mol_objs, folds, labels =_load_bace_dataset(input_path)

    data_list, data_smiles_list = [], []
    for i in range(len(smiles_list)):
        rdkit_mol = rdkit_mol_objs[i]
        # # convert aromatic bonds to double bonds
        # Chem.SanitizeMol(rdkit_mol,
        #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        data, smiles, err = mol_to_graph_data_obj_simple(rdkit_mol)
        if data == None: 
            print("{} is removed".format(smiles_list[i]))
            continue ## addition
        # manually add mol id
        data.id = torch.tensor(
            [i])  # id here is the index of the mol in
        # the dataset
        data.y = torch.tensor([labels[i]])
        data.fold = torch.tensor([folds[i]])
        data_list.append(data)
        data_smiles_list.append(smiles)
    
    return data_list, data_smiles_list


def _load_bace_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array
    containing indices for each of the 3 folds, np.array containing the
    labels
    """
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['mol']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['Class']
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # there are no nans
    folds = input_df['Model']
    folds = folds.replace('Train', 0)   # 0 -> train
    folds = folds.replace('Valid', 1)   # 1 -> valid
    folds = folds.replace('Test', 2)    # 2 -> test
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    assert len(smiles_list) == len(folds)
    return smiles_list, rdkit_mol_objs_list, folds.values, labels.values
