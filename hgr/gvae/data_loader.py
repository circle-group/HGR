import os.path
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from hgr.utils.file_utils import PathManager
# from hgr.utils.debug_utils import Timer
import ringdiv


def _sync_filter_none(rules, targets=None, smiles=None, split="train"):
    """
    guacamol train中有3个None表示分子解析失败了，需要剔除None
    """
    if targets is None and smiles is None:
        kept = [r for r in rules if r is not None]
        removed = len(rules) - len(kept)
        if removed:
            print(f"[Warn] {split}: removed {removed} None rules")
        return kept, None, None

    # 统一成可 zip 的序列
    no_targets, no_smiles = False, False
    if targets is None:
        targets = [None] * len(rules)
        no_targets = True
    if smiles is None:
        smiles = [None] * len(rules)
        no_smiles = True

    kept_rules, kept_targets, kept_smiles = [], [], []
    removed = 0
    for r, y, s in zip(rules, targets, smiles):
        if r is None:
            removed += 1
            continue
        kept_rules.append(r)
        kept_targets.append(y)
        kept_smiles.append(s)

    if removed:
        print(f"[Warn] {split}: removed {removed} None rules")

    # 如果原本 targets/smiles 是 None，就保持 None
    out_targets = None if no_targets else kept_targets
    out_smiles = None if no_smiles else kept_smiles
    return kept_rules, out_targets, out_smiles

def get_dataloaders(config, prod_rule_seq_list, num_prod_rule, target_val_list=None, smile_list=None):
    data_config = config.data
    bs = config.gvae.bs
    max_len = data_config.max_vol
    data_name = data_config.name.lower()

    # -------------- 1) 统一 train/test 数据源 --------------
    # 模式A：prod_rule_seq_list 是 list（QM9/ZINC），需要用 valid_idx 切分
    # 模式B：prod_rule_seq_list 是 dict，里面有 ['train'], ['test']（MOSES/GuacaMol/RingDiv），已切好
    if isinstance(prod_rule_seq_list, dict):
        assert data_name in ['moses', 'guacamol'], \
            "When prod_rule_seq_list is a dict, data_name should be in ['moses', 'guacamol']."
        assert "train" in prod_rule_seq_list and "test" in prod_rule_seq_list, \
            "When prod_rule_seq_list is a dict, it must contain keys: 'train' and 'test'."
        train_rules_raw = prod_rule_seq_list["train"]
        test_rules_raw = prod_rule_seq_list["test"]

        targetV_train, targetV_test = None, None
        if isinstance(target_val_list, dict):
            targetV_train = target_val_list.get("train", None)
            targetV_test = target_val_list.get("test", None)
            assert len(targetV_train) == len(train_rules_raw)
            assert len(targetV_test) == len(test_rules_raw)
            
        smiles_train, smiles_test = None, None
        if isinstance(smile_list, dict):
            smiles_train = smile_list.get("train", None)
            smiles_test = smile_list.get("test", None)
            assert len(smiles_train) == len(train_rules_raw)
            assert len(smiles_test) == len(test_rules_raw)

        if data_name == 'guacamol':
            train_rules_raw, targetV_train, smiles_train = _sync_filter_none(train_rules_raw, targetV_train, smiles_train, split="train")
            # test_rules_raw, targetV_test, smiles_test = _sync_filter_none(test_rules_raw, targetV_test, smiles_test, split="test")
            
    else:
        # list 模式（QM9/ZINC250k）：用 valid_idx 切分逻辑
        assert data_name in ['qm9', 'zinc250k', 'ringdiv', 'ringdiv300k'], \
            "When prod_rule_seq_list is a list, data_name should be in ['qm9', 'zinc250k', 'ringdiv300k', 'ringdiv']."
        N = len(prod_rule_seq_list)
        
        if data_name in ['qm9', 'zinc250k']:
            with open(os.path.join(PathManager.DATA_DIR, config.path.valid_idx)) as f:
                test_idx = json.load(f)
            if data_name == 'qm9':
                test_idx = list(map(int, test_idx['valid_idxs']))
            test_idx = np.asarray(test_idx, dtype=np.int64)
        elif data_name in ['ringdiv300k', 'ringdiv']:
            # ringdiv/ringdiv300k：从 datasets/{name}/raw/{name}_test_idx.npz 读取
            # raw_dir = dataset_raw_dir(data_name, data_root=PathManager.DATA_ROOT)
            test_idx_npz_path = os.path.join(PathManager.DATA_DIR, 'raw', f"{data_name}_test_idx.npz")
            test_idx, meta = ringdiv.read_test_idx(test_idx_npz_path)

            # 可选一致性检查（如果 npz 里写了 n_total）
            if "n_total" in meta and int(meta["n_total"]) != N:
                raise ValueError(f"n_total mismatch: npz={meta['n_total']} vs prod_rule_seq_list={N}")

        # 越界检查
        if test_idx.size > 0 and (test_idx.min() < 0 or test_idx.max() >= N):
            raise ValueError(f"test_idx out of range: min={test_idx.min()}, max={test_idx.max()}, N={N}")

        # 2) mask 保序切分
        mask = np.zeros(N, dtype=bool)
        mask[test_idx] = True
        test_pos, train_pos = np.where(mask)[0], np.where(~mask)[0]

        train_rules_raw = [prod_rule_seq_list[i] for i in train_pos]
        test_rules_raw = [prod_rule_seq_list[i] for i in test_pos]

        targetV_train, targetV_test = None, None
        if target_val_list is not None:
            assert len(target_val_list) == N
            targetV_train = target_val_list[train_pos]
            targetV_test = target_val_list[test_pos]
        
        smiles_train, smiles_test = None, None
        if smile_list is not None:
            assert len(smile_list) == N
            smiles_train = [smile_list[i] for i in train_pos]
            smiles_test = [smile_list[i] for i in test_pos]


    # -------------- 2) padding（train/test 一起算最大长度） --------------
    all_lens = [len(s) for s in train_rules_raw] + [len(s) for s in test_rules_raw]
    actual_max_rule_len = max(all_lens) if len(all_lens) > 0 else 0
    print(f"[Info] actual_rule_len = {actual_max_rule_len}, max_len = {max_len}")
    assert max_len >= actual_max_rule_len, f"`max_len` should be larger than the maximum length of input rule sequences, {actual_max_rule_len}."

    pad_idx = config.data.padding_idx % (num_prod_rule + 1) # 最后一个 id 预留为 padding

    # def pad_and_flip(rule_seqs):
    #     # 我们的规则从右往左运用，MHG的是从左往右
    #     left_pad = torch.stack(
    #         [torch.LongTensor([pad_idx] * (max_len - len(s)) + s) for s in rule_seqs],
    #         dim=0
    #     )
    #     right_pad = torch.flip(left_pad, dims=[1])
    #     return left_pad, right_pad # shape: (N, max_len)
    def pad_and_flip(rule_seqs):
        # 相比上面的版本更快（处理guacamol 从21s降到13s）
        left_pad = torch.full((len(rule_seqs), max_len), pad_idx, dtype=torch.long)
        for i, s in enumerate(rule_seqs):
            s = torch.as_tensor(s, dtype=torch.long)
            left_pad[i, max_len - len(s):] = s
        right_pad = torch.flip(left_pad, dims=[1])
        return left_pad, right_pad

    ruleL_train, ruleR_train = pad_and_flip(train_rules_raw)
    ruleL_test, ruleR_test = pad_and_flip(test_rules_raw)

    # -------------- 3) dataset / dataloader --------------
    train_ds = HGRDataset(ruleL_train, ruleR_train, target_val_list=targetV_train, smile_list=smiles_train)
    test_ds = HGRDataset(ruleL_test, ruleR_test, target_val_list=targetV_test, smile_list=smiles_test)

    num_workers = int(
        getattr(
            getattr(config, "train", None),
            "dataloader_workers",
            getattr(getattr(config, "data", None), "dataloader_workers", 8),
        )
    )
    pin_memory = bool(getattr(getattr(config, "train", None), "pin_memory", True))
    common_kwargs = {
        "batch_size": config.gvae.bs,
        "drop_last": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        common_kwargs["persistent_workers"] = bool(
            getattr(getattr(config, "train", None), "persistent_workers", True)
        )
    hgr_train_loader = DataLoader(train_ds, shuffle=True, **common_kwargs)
    hgr_test_loader = DataLoader(test_ds, shuffle=False, **common_kwargs)

    return hgr_train_loader, hgr_test_loader



class HGRDataset(Dataset):
    '''
    A class of HRG data
    '''

    def __init__(self, left_padded_rules, right_padded_rules, target_val_list=None, smile_list=None):
        #self.hrg = hrg
        # 我们的规则从右往左运用，MHG的是从左往右

        self.left_padded_rules = left_padded_rules
        self.right_padded_rules = right_padded_rules

        self.target_val_list = target_val_list
        self.smile_list = smile_list

    def __len__(self):
        return len(self.left_padded_rules)

    def __getitem__(self, idx):
        # 暂时只支持这一种
        if self.smile_list is not None:
            return  self.left_padded_rules[idx], self.right_padded_rules[idx], self.smile_list[idx]

        return self.left_padded_rules[idx], self.right_padded_rules[idx]







# def get_dataloaders_v1(data_config, prod_rule_seq_list, target_val_list=None, smile_list=None, shuffle=False):
#     ''' return a dataloader for train/val/test

#         Parameters
#         ----------
#         prod_rule_seq_list : List of lists
#             each element corresponds to a sequence of production rules.
#             生成规则序列，每个元素对应一个分子的规则序列。  # 注意：我们的规则从右往左运用，MHG的是从左往右
#         train_params : dict
#             self.Train_params

#         Returns
#         -------
#         Dataloaders for train, val, test of autoencoders
#             each batch contains two torch Tensors, each of which corresponds to input and output of autoencoder.
#     '''

#     bs = data_config.bs
#     max_len = data_config.max_vol
#     split_sizes = (data_config.train_size, data_config.val_size, data_config.test_size)

#     actual_rule_len = max([len(each_rule) for each_rule in prod_rule_seq_list])
#     print(f"[Info] actual_rule_len = {actual_rule_len}, max_len = {max_len}")
#     assert max_len >= actual_rule_len, f"`max_len` should be larger than the maximum length of input rule sequences, {actual_rule_len}."

#     def split_list(lst, split_sizes):
#         """按 split_sizes 切分 lst，split_sizes=(train, val, test)"""
#         train_size, val_size, test_size = split_sizes
#         return lst[:train_size], lst[train_size:train_size + val_size], lst[train_size + val_size:train_size + val_size + test_size]

#     # 划分训练、验证、测试集
#     rule_train, rule_val, rule_test = split_list(prod_rule_seq_list, split_sizes)
#     targetV_train, targetV_val, targetV_test = split_list(target_val_list, split_sizes) if target_val_list is not None else (None, )*3
#     smiles_train, smiles_val, smiles_test = split_list(smile_list, split_sizes) if smile_list is not None else (None, )*3

#     def create_dataloader(data, targets, smile_list=None):
#         return DataLoader(
#             dataset=HGRDataset(data, max_len, target_val_list=targets, smile_list = smile_list),
#             batch_size=bs, shuffle=shuffle, drop_last=False,
#         )

#     hgr_train_loader = create_dataloader(rule_train, targetV_train, smiles_train)
#     hgr_label_loader = create_dataloader(rule_val, targetV_val, smiles_val)
#     hgr_test_loader = create_dataloader(rule_test, targetV_test, smiles_test)

#     return hgr_train_loader, hgr_label_loader, hgr_test_loader

# def get_dataloaders_v2(config, prod_rule_seq_list, num_prod_rule, target_val_list=None, smile_list=None, shuffle=False):
#     ''' 
#     这个版本只支持qm9和zinc250k, 不支持moses和guacamol，因此更新到v3版本
#     return a dataloader for train/val/test

#         Parameters
#         ----------
#         prod_rule_seq_list : List of lists
#             each element corresponds to a sequence of production rules.
#             生成规则序列，每个元素对应一个分子的规则序列。  # 注意：HGR的规则从右往左运用，MHG的是从左往右
#         train_params : dict
#             self.Train_params

#         Returns
#         -------
#         Dataloaders for train, val, test of autoencoders
#             each batch contains two torch Tensors, each of which corresponds to input and output of autoencoder.
#     '''
#     # ——配置参数 and 参数校验——
#     data_config = config.data
#     bs = config.gvae.bs
#     max_len = data_config.max_vol
#     N = len(prod_rule_seq_list)
#     if target_val_list is not None:
#         assert len(target_val_list) == N, f"Length of target_val_list ({len(target_val_list)}) should be equal to the length of prod_rule_seq_list ({N})."
#     if smile_list is not None:
#         assert len(smile_list) == N, f"Length of smile_list ({len(smile_list)}) should be equal to the length of prod_rule_seq_list ({N})."

#     actual_rule_len = max([len(each_rule) for each_rule in prod_rule_seq_list])
#     print(f"[Info] actual_rule_len = {actual_rule_len}, max_len = {max_len}")
#     assert max_len >= actual_rule_len, f"`max_len` should be larger than the maximum length of input rule sequences, {actual_rule_len}."

#     pad_idx = config.data.padding_idx % (num_prod_rule + 1)  # 最后一个 id 预留为 padding
#     # 我们的规则从右往左运用，MHG的是从左往右
#     left_padded_rules = torch.stack([torch.LongTensor([pad_idx] * (max_len - len(s)) + s)
#         for s in prod_rule_seq_list], dim=0)  # (N, max_len)
#     right_padded_rules = torch.flip(left_padded_rules, dims=[1])  # (N, max_len)

#     with open(os.path.join(PathManager.DATA_DIR, config.path.valid_idx)) as f:
#         test_idx = json.load(f)
#     if config.data.name == 'qm9':
#         test_idx = list(map(int, test_idx['valid_idxs']))
#     train_idx = list(set(range(N)) - set(test_idx))

#     # 划分训练、验证、测试集
#     ruleL_train, ruleL_test = left_padded_rules[train_idx], left_padded_rules[test_idx]
#     ruleR_train, ruleR_test = right_padded_rules[train_idx], right_padded_rules[test_idx]
#     targetV_train, targetV_test = (target_val_list[train_idx], target_val_list[test_idx]) if target_val_list is not None else (None, None)
#     # smiles_train, smiles_test = (smile_list[train_idx], smile_list[test_idx]) if smile_list is not None else (None, None)
#     if smile_list is not None:
#         smiles_train = [smile_list[i] for i in train_idx]
#         smiles_test = [smile_list[i] for i in test_idx]
#     else:
#         smiles_train, smiles_test = None, None

#     # def create_dataloader(ruleL, ruleR, targets, smile_list=None):
#     #     return DataLoader(
#     #         dataset=HRGDataset(ruleL, ruleR, target_val_list=targets, smile_list = smile_list),
#     #         batch_size=bs, shuffle=shuffle, drop_last=False,num_workers=8
#     #     )

#     # hrg_dataloader_train = create_dataloader(ruleL_train, ruleR_train, targetV_train, smiles_train)
#     # hrg_dataloader_test = create_dataloader(ruleL_test, ruleR_test, targetV_test, smiles_test)

#     train_ds = HGRDataset(ruleL_train, ruleR_train, target_val_list=targetV_train, smile_list = smiles_train)
#     test_ds = HGRDataset(ruleL_test, ruleR_test, target_val_list=targetV_test, smile_list = smiles_test)

#     hgr_train_loader = DataLoader(train_ds, batch_size=bs, drop_last=False, num_workers=8, shuffle=True)
#     hgr_test_loader = DataLoader(test_ds, batch_size=bs, drop_last=False, num_workers=8, shuffle=False)

#     return hgr_train_loader, hgr_test_loader
