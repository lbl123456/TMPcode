"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn
import copy


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.t()).float().to(device)
            Negative_mask = torch.ne(labels, labels.t()).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))



        #Sim 分子计算
        dot_contrast = torch.matmul(anchor_feature, anchor_feature.t())
        dot_contrast_copy = dot_contrast.clone()
        norm_2 = []
        for i in range(batch_size*2):         
            v1 = torch.norm(anchor_feature[i],2)
            norm_2.append(v1)
        for i in range(batch_size*2):
            for j in range(batch_size*2):
                #除去分母
                dot_contrast[i][j] = dot_contrast_copy[i][j] / (norm_2[i] * norm_2[j])
        #除去T
        logits = torch.div(dot_contrast,self.temperature)


        
        mask = mask.repeat(anchor_count, contrast_count)   #自己的class
        Negative_mask = Negative_mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # 本ID之外的，也就是负样本
        exp_logits = torch.exp(logits) * Negative_mask   

        #分母是再加上自己（不加正样本）
        v1 = exp_logits.sum(1, keepdim=True)
        v2 = v1.repeat(1,batch_size*2)
        v2 = v2.t()
        v2 = v2 + torch.exp(logits)
        #v3即每一次计算的分母矩阵
        v3 = torch.log(v2)
        #log相减
        log_prob = logits - v3

        #相加
        mean_log_prob_pos = (mask * log_prob).sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
