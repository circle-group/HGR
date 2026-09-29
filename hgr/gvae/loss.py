#!/usr/bin/env python
"""
优化记录：
GrammarVAELoss_v1 -> GrammarVAELoss_v2: 将循环改为了矩阵操作，大幅提升训练效率
GrammarVAELoss_v2 -> GrammarVAELoss: 增加了CyclicalBetaScheduler
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class VAELoss(_Loss):
    '''
    a loss function for VAE
    '''

    def __init__(self, ignore_index=None, beta=1.0, **kwargs):
        super().__init__(**kwargs)
        self.ignore_index = ignore_index
        self.beta = beta

    def forward(self, in_seq_pred, in_seq, mu, logvar):
        ''' compute VAE loss

        Parameters
        ----------
        in_seq_pred : torch.Tensor, shape (batch_size, max_len, vocab_size)
            logit
        in_seq : torch.Tensor, shape (batch_size, max_len)
            each element corresponds to a word id in vocabulary.
        mu : torch.Tensor, shape (batch_size, hidden_dim)
        logvar : torch.Tensor, shape (batch_size, hidden_dim)
            mean and log variance of the normal distribution
        '''
        cross_entropy = F.cross_entropy(
            in_seq_pred.view(-1, in_seq_pred.shape[2]),
            in_seq.view(-1),
            reduction='sum',
            ignore_index=self.ignore_index if self.ignore_index is not None else -100)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return cross_entropy + self.beta * kl_div



# class GrammarVAELoss_v1(_Loss):
#     ''' 优化后版本为GrammarVAELoss， qm9 一个epoch 106s->13s
#     a loss function for Grammar VAE

#     Attributes
#     ----------
#     hrg : HyperedgeReplacementGrammar
#     ignore_index : int
#         index to be ignored
#     beta : float
#         coefficient of KL divergence
#     '''

#     def __init__(self, prod_rule_corpus, ignore_index=None, beta=0.01, class_weight=None, **kwargs):
#         super().__init__(**kwargs)
#         self.beta = beta
#         self.prod_rule_corpus = prod_rule_corpus
#         self.class_weight = class_weight

#         num_pr = prod_rule_corpus.num_prod_rule
#         vocab_size = num_pr + 1

#         self.ignore_index = int(np.mod(-1, vocab_size))



#     def forward(self, in_seq_pred, in_seq, mu, logvar, beta=None):
#         ''' compute VAE loss

#         Parameters
#         ----------
#         in_seq_pred : torch.Tensor, shape (batch_size, max_len, vocab_size)
#             logit
#         in_seq : torch.Tensor, shape (batch_size, max_len)
#             each element corresponds to a word id in vocabulary.
#         mu : torch.Tensor, shape (batch_size, hidden_dim)
#         logvar : torch.Tensor, shape (batch_size, hidden_dim)
#             mean and log variance of the normal distribution
#         '''
#         if beta is None:
#             beta = self.beta

#         batch_size, max_len, vocab_size = in_seq_pred.shape
#         self.ignore_index = int(np.mod(-1, vocab_size))

#         mask = torch.zeros_like(in_seq_pred)

#         for each_batch in range(batch_size):
#             for each_idx in range(max_len):
#                 prod_rule_idx = in_seq[each_batch, each_idx]
#                 if prod_rule_idx == self.ignore_index:
#                     continue
#                 nt_sym = self.prod_rule_corpus.prod_rule_list[prod_rule_idx].lhs_nt_symbol
#                 nt_idx = self.prod_rule_corpus._nt_symbol_to_idx[nt_sym]
#                 mask[each_batch, each_idx, :-1] = self.prod_rule_corpus.lhs_in_prod_rule[nt_idx]
#         #mask = mask.to(in_seq_pred.device)
#         in_seq_pred = mask * in_seq_pred

#         cross_entropy = F.cross_entropy(
#             in_seq_pred.view(-1, vocab_size),
#             in_seq.view(-1),
#             weight=self.class_weight,
#             reduction='sum',
#             ignore_index=self.ignore_index if self.ignore_index is not None else -100)
#         kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#         return cross_entropy + beta * kl_div, cross_entropy, kl_div




# class GrammarVAELoss_v2(_Loss):
#     """
#     An optimized loss function for Grammar VAE with detailed comments.

#     Attributes
#     ----------
#     prod_rule_corpus : ProductionRuleCorpus
#         Contains all production rules and grammar masks.
#     beta : float
#         Weight for the KL divergence term.
#     class_weight : torch.Tensor or None
#         Optional weights for the cross-entropy loss.
#     ignore_index : int
#         Index used to pad sequences and ignore in loss calculation.
#     prod_rule_mask : torch.Tensor (buffer)
#         A precomputed boolean mask of shape (num_pr+1, vocab_size) that indicates
#         which production rules are legal for each LHS non-terminal (and one extra
#         row for PAD/ignore_index).
#     """

#     def __init__(self, prod_rule_corpus, beta=0.01, class_weight=None):
#         super().__init__()
#         self.prod_rule_corpus = prod_rule_corpus
#         self.beta = beta
#         self.class_weight = class_weight

#         # Determine number of production rules and vocabulary size (+1 for PAD)
#         num_pr = prod_rule_corpus.num_prod_rule
#         vocab_size = num_pr + 1
#         self.ignore_index = int(np.mod(-1, vocab_size)) # Use the last index as ignore_index (PAD token)

#         # --------------------------------------------------------------------
#         # Build the prod_rule_mask buffer:
#         # shape = (num_pr+1, vocab_size)
#         # --------------------------------------------------------------------

#         # 1) Map each production rule index j -> its LHS non-terminal index
#         # prod_to_lhs：一个长度为 num_pr 的 LongTensor，
#         # prod_to_lhs[j] = 该产生式 j 在 nt_symbol_list 里的行索引
#         prod_to_lhs = torch.tensor(
#             [prod_rule_corpus._nt_symbol_to_idx[pr.lhs_nt_symbol]
#              for pr in prod_rule_corpus.prod_rule_list],
#             dtype=torch.long
#         )  # shape: (num_pr,)

#         # 2) Get the dense grammar mask: shape (num_nt_symbols, num_pr)
#         # lhs_in_prod_rule 本来是一个 (num_nt_symbols, num_prod_rules) 的 dense Tensor
#         lhs_mask = prod_rule_corpus.lhs_in_prod_rule.bool() # (max_len, num_pr)

#         # 3) Gather rows for each production rule: shape (num_pr, num_pr)
#         #    Row j = lhs_mask[ prod_to_lhs[j] ]
#         # 3) 用 prod_to_lhs 去 gather 出 (num_pr, num_pr) 的 “行掩码矩阵”
#         #    第 j 行就是 `lhs_mask[ prod_to_lhs[j] ]`
#         prod_rule_mask = lhs_mask[prod_to_lhs] # shape (num_pr, num_pr)

#         # 4) Append a column of False for the PAD token in predictions
#         pad_col = torch.zeros((num_pr, 1), dtype=torch.bool)
#         prod_rule_mask = torch.cat([prod_rule_mask, pad_col], dim=1)  # -> (num_pr, num_pr+1)

#         # 5) Append a row of False for ignore_index to handle PAD rows safely
#         pad_row = torch.zeros((1, vocab_size), dtype=torch.bool)
#         prod_rule_mask = torch.cat([prod_rule_mask, pad_row], dim=0)  # -> (num_pr+1, num_pr+1)

#         # 6) Register as buffer so it moves with model.to(device)
#         self.register_buffer('prod_rule_mask', prod_rule_mask)

#     def forward(self, in_seq_pred, in_seq, mu, logvar, beta=None):
#         """
#         Compute the VAE loss with grammar constraints applied to logits.

#         Parameters
#         ----------
#         in_seq_pred : torch.Tensor, shape (batch_size, max_len, vocab_size)
#             Decoder logits before softmax.
#         in_seq : torch.LongTensor, shape (batch_size, max_len)
#             Ground-truth sequence of production rule indices (PAD = ignore_index).
#         mu : torch.Tensor, shape (batch_size, hidden_dim)
#             Latent mean from encoder.
#         logvar : torch.Tensor, shape (batch_size, hidden_dim)
#             Latent log-variance from encoder.
#         beta : float or None
#             Weight for KL divergence term (defaults to self.beta).

#         Returns
#         -------
#         total_loss : torch.Tensor
#             Sum of reconstruction loss and weighted KL divergence.
#         recon_loss : torch.Tensor
#             Reconstruction (cross-entropy) loss sum.
#         kl_div : torch.Tensor
#             KL divergence term sum.
#         """
#         # Use provided beta or fallback to default
#         if beta is None: beta = self.beta

#         # Shapes
#         batch_size, max_len, vocab_size = in_seq_pred.shape

#         # Fetch the boolean mask for each position: shape (batch_size, max_len, vocab_size)
#         # For ignore_index rows, this returns the last row of prod_rule_mask (all False)
#         # 用 fancy-indexing 一次性取出 [B, L, vocab_size]
#         #  注意：padding 位（== ignore_index）会被当成 “索引超界” ,指到最后一行（全 0），
#         mask = self.prod_rule_mask[in_seq]

#         # Apply mask by setting illegal logits to -inf
#         # This ensures softmax assigns zero probability to illegal productions
#         in_seq_pred = in_seq_pred.masked_fill(~mask, float('-inf'))
#         # in_seq_pred = in_seq_pred * mask # 旧版本是乘0


#         # Reconstruction loss: ignore PAD tokens via ignore_index
#         recon_loss = F.cross_entropy(
#             in_seq_pred.view(-1, vocab_size),
#             in_seq.view(-1),
#             weight=self.class_weight,
#             ignore_index=self.ignore_index,
#             reduction='sum'
#         )

#         # KL divergence term
#         kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

#         # Total VAE loss
#         total_loss = recon_loss + beta * kl_div
#         return total_loss, recon_loss, kl_div


class CyclicalBetaScheduler:
    """
    Cyclical annealing scheduler for beta in VAE loss using explicit phase lengths.

    Args:
        anneal_steps (int): Number of steps to anneal beta from 0 to 1 in each cycle.
        fix_steps (int): Number of steps to keep beta at 1 in each cycle.
    """
    def __init__(self, anneal_steps: int, fix_steps: int):
        self.anneal_steps = anneal_steps
        self.fix_steps = fix_steps
        self.cycle_length = anneal_steps + fix_steps

    def get_beta(self, step: int) -> float:
        """Get beta value for the given global step."""
        # map step into its position in the cycle
        cycle_step = step % self.cycle_length
        # annealing phase
        if cycle_step < self.anneal_steps:
            return cycle_step / float(self.anneal_steps)
        # fixing phase
        return 1.0


class GrammarVAELoss(_Loss):
    """
    An optimized loss function for Grammar VAE.

    Attributes
    ----------
    prod_rule_corpus : ProductionRuleCorpus
        Contains all production rules and grammar masks.
    beta : float
        Weight for the KL divergence term.
    class_weight : torch.Tensor or None
        Optional weights for the cross-entropy loss.
    ignore_index : int
        Index used to pad sequences and ignore in loss calculation.
    prod_rule_mask : torch.Tensor (buffer)
        A precomputed boolean mask of shape (num_pr+1, vocab_size) that indicates
        which production rules are legal for each LHS non-terminal (and one extra
        row for PAD/ignore_index).
    """

    def __init__(self, prod_rule_corpus, anneal_steps, fix_steps, beta=0.01,  class_weight=None):
        super().__init__()
        self.prod_rule_corpus = prod_rule_corpus
        self.beta = beta
        self.class_weight = class_weight

        # Determine number of production rules and vocabulary size (+1 for PAD)
        num_pr = prod_rule_corpus.num_prod_rule
        vocab_size = num_pr + 1
        self.ignore_index = int(np.mod(-1, vocab_size)) # Use the last index as ignore_index (PAD token)

        # --------------------------------------------------------------------
        # Build the prod_rule_mask buffer:
        # shape = (num_pr+1, vocab_size)
        # --------------------------------------------------------------------

        # 1) Map each production rule index j -> its LHS non-terminal index
        # prod_to_lhs：一个长度为 num_pr 的 LongTensor，
        # prod_to_lhs[j] = 该产生式 j 在 nt_symbol_list 里的行索引
        prod_to_lhs = torch.tensor(
            [prod_rule_corpus._nt_symbol_to_idx[pr.lhs_nt_symbol]
             for pr in prod_rule_corpus.prod_rule_list],
            dtype=torch.long
        )  # shape: (num_pr,)

        # 2) Get the dense grammar mask: shape (num_nt_symbols, num_pr)
        # lhs_in_prod_rule 本来是一个 (num_nt_symbols, num_prod_rules) 的 dense Tensor
        lhs_mask = prod_rule_corpus.lhs_in_prod_rule.bool() # (max_len, num_pr)

        # 3) Gather rows for each production rule: shape (num_pr, num_pr)
        #    Row j = lhs_mask[ prod_to_lhs[j] ]
        # 3) 用 prod_to_lhs 去 gather 出 (num_pr, num_pr) 的 “行掩码矩阵”
        #    第 j 行就是 `lhs_mask[ prod_to_lhs[j] ]`
        prod_rule_mask = lhs_mask[prod_to_lhs] # shape (num_pr, num_pr)

        # 4) Append a column of False for the PAD token in predictions
        pad_col = torch.zeros((num_pr, 1), dtype=torch.bool)
        prod_rule_mask = torch.cat([prod_rule_mask, pad_col], dim=1)  # -> (num_pr, num_pr+1)

        # 5) Append a row of False for ignore_index to handle PAD rows safely
        pad_row = torch.zeros((1, vocab_size), dtype=torch.bool)
        prod_rule_mask = torch.cat([prod_rule_mask, pad_row], dim=0)  # -> (num_pr+1, num_pr+1)

        # 6) Register as buffer so it moves with model.to(device)
        self.register_buffer('prod_rule_mask', prod_rule_mask)

        # setup cyclical scheduler with explicit phase lengths
        self.beta_scheduler = CyclicalBetaScheduler(
            anneal_steps=anneal_steps,
            fix_steps=fix_steps
        )

    def forward(self, in_seq_pred, in_seq, mu, logvar, step=None):
        """
        Compute the VAE loss with grammar constraints applied to logits.

        Parameters
        ----------
        in_seq_pred : torch.Tensor, shape (batch_size, max_len, vocab_size)
            Decoder logits before softmax.
        in_seq : torch.LongTensor, shape (batch_size, max_len)
            Ground-truth sequence of production rule indices (PAD = ignore_index).
        mu : torch.Tensor, shape (batch_size, hidden_dim)
            Latent mean from encoder.
        logvar : torch.Tensor, shape (batch_size, hidden_dim)
            Latent log-variance from encoder.

        Returns
        -------
        total_loss : torch.Tensor
            Sum of reconstruction loss and weighted KL divergence.
        recon_loss : torch.Tensor
            Reconstruction (cross-entropy) loss sum.
        kl_div : torch.Tensor
            KL divergence term sum.
        """
        # Use provided beta or fallback to default
        if step is not None:
            beta = self.beta * self.beta_scheduler.get_beta(step)
        else:
            beta = self.beta

        # Shapes
        batch_size, max_len, vocab_size = in_seq_pred.shape

        # Fetch the boolean mask for each position: shape (batch_size, max_len, vocab_size)
        # For ignore_index rows, this returns the last row of prod_rule_mask (all False)
        # 用 fancy-indexing 一次性取出 [B, L, vocab_size]
        #  注意：padding 位（== ignore_index）会被当成 “索引超界” ,指到最后一行（全 0），
        mask = self.prod_rule_mask[in_seq] #含义：对 batch 中每个样本、每个rule，给出一个长度为 321 的布尔向量，表示该位置“允许预测哪些 token”

        # Apply mask by setting illegal logits to -inf
        # This ensures softmax assigns zero probability to illegal productions
        in_seq_pred = in_seq_pred.masked_fill(~mask, float('-inf'))
        # in_seq_pred = in_seq_pred * mask # 旧版本是乘0


        # Reconstruction loss: ignore PAD tokens via ignore_index
        recon_loss = F.cross_entropy(
            in_seq_pred.view(-1, vocab_size),
            in_seq.view(-1),
            weight=self.class_weight,
            ignore_index=self.ignore_index,
            reduction='sum'
        )

        # KL divergence term
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Total VAE loss
        total_loss = recon_loss + beta * kl_div
        return total_loss, recon_loss, kl_div
