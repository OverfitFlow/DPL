import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from clip.modules import objectives


class DecoupledType3(nn.Module):
    """
    simple decouple missing and non-missing class wise prototypes
    """
    
    def __init__(self, n_prototypes, n_hidden, zerout=False):
        super().__init__()
        
        self.n_prototypes = n_prototypes
        self.n_hidden = n_hidden
        self.zerout = zerout
        
        # w_all = []
        
        # w_c = torch.empty(n_hidden, n_prototypes)
        # nn.init.xavier_normal_(w_c)
        # w_all.append(w_c)
        
        # w_tm = torch.empty(n_hidden, n_prototypes)
        # nn.init.xavier_normal_(w_tm)
        # w_all.append(w_tm)
        
        # w_im = torch.empty(n_hidden, n_prototypes)
        # nn.init.xavier_normal_(w_im)
        # w_all.append(w_im)
        
        # self.w = nn.Parameter(torch.stack(w_all, dim=0), requires_grad=True)

        w_mi_c = torch.empty(n_hidden//2, n_prototypes)
        nn.init.xavier_normal_(w_mi_c)
        self.w_mi_c = nn.Parameter(w_mi_c)
        
        w_mi_tm = torch.empty(n_hidden//2, n_prototypes)
        nn.init.xavier_normal_(w_mi_tm)
        self.w_mi_tm = nn.Parameter(w_mi_tm)

        w_mt_c = torch.empty(n_hidden//2, n_prototypes) # text modality prototype when image is complete
        nn.init.xavier_normal_(w_mt_c)
        self.w_mt_c = nn.Parameter(w_mt_c)
        
        w_mt_im = torch.empty(n_hidden//2, n_prototypes) # text modality prototype when image is missing
        nn.init.xavier_normal_(w_mt_im)
        self.w_mt_im = nn.Parameter(w_mt_im)

        if not self.zerout:
            w_mi_im = torch.empty(n_hidden//2, n_prototypes)
            nn.init.xavier_normal_(w_mi_im)
            w_mt_tm = torch.empty(n_hidden//2, n_prototypes) # text modality prototype when image is missing
            nn.init.xavier_normal_(w_mt_tm)
            self.w_mi_im = nn.Parameter(w_mi_im)
            self.w_mt_tm = nn.Parameter(w_mt_tm)
            
        else:
            w_mi_im = torch.zeros(n_hidden//2, n_prototypes)
            w_mt_tm = torch.zeros(n_hidden//2, n_prototypes)
            self.w_mi_im = nn.Parameter(w_mi_im, requires_grad=False)
            self.w_mt_tm = nn.Parameter(w_mt_tm, requires_grad=False)
        
    def get_prototypes(self):
        # if self.zerout:
        #     mul = torch.ones_like(self.w)
        #     feature_dim = self.n_hidden // 2
        #     mul[1, feature_dim:].zero_()
        #     mul[2, :feature_dim].zero_()
        #     w = self.w * mul
        # else:
        #     w = self.w
        w = torch.stack([
                torch.cat([self.w_mi_c, self.w_mt_c], dim=0),
                torch.cat([self.w_mi_tm, self.w_mt_tm], dim=0),
                torch.cat([self.w_mi_im, self.w_mt_im], dim=0),
        ], dim=0)
        return F.normalize(w, dim=1)
        
    def forward(self, x, missing_type):
        
        inp = F.normalize(x, dim=1)
        
        w = self.get_prototypes()
        w_select = w[missing_type]
        
        logits =  torch.matmul(inp.unsqueeze(1), w_select).squeeze(1)  # bs x n_prototypes
        
        return logits


class ArcMarginProduct(nn.Module):
    """
    https://github.com/egcode/pytorch-losses/blob/master/mnist-visualize-arcface-loss.ipynb
    """
    def __init__(self, s=64.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        # self.in_features = in_features
        # self.out_features = out_features
        
        self.s = s
        self.m = m
        
        # self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        # nn.init.xavier_uniform_(self.weight)
        # self.device = device

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, logits, label):
        # self.cos_m = torch.tensor(self.cos_m, dtype=logits.dtype)
        # self.sin_m = torch.tensor(self.sin_m, dtype=logits.dtype)
        # self.th = torch.tensor(self.th, dtype=logits.dtype)
        # self.mm = torch.tensor(self.mm, dtype=logits.dtype)

        # self.cos_m, self.sin_m, self.th, self.mm = self.cos_m.to(logits.dtype), self.sin_m.to(logits.dtype), self.th.to(logits.dtype), self.mm.to(logits.dtype)
        
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = logits.float()
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device=logits.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output


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
            mask = torch.eq(labels, labels.T).float().to(device)
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

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, torch.ones_like(mask_pos_pairs), mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


def compute_food101_plt3(pl_module, batch):
    phase = "train" if pl_module.training else "val"
    if phase == "train":
        infer = pl_module.infer(batch)
    else:
        infer = pl_module.infer(batch)

    clf_feats = pl_module.food101_proj(infer["cls_feats"])
    clf_labels = torch.tensor(batch["label"]).to(pl_module.device).long()
    
    m = torch.tensor(batch["missing_type"], device=pl_module.device).long()
    clf_logits = pl_module.food101_pl_head(clf_feats, m)
    
    clf_logits_margin = torch.where(
        m.reshape(-1, 1) == 0, pl_module.food101_arc_head(clf_logits, clf_labels),
        torch.where(m.reshape(-1, 1) == 1, pl_module.food101_arc_head_tm(clf_logits, clf_labels), pl_module.food101_arc_head_im(clf_logits, clf_labels))
     )
    clf_loss = F.cross_entropy(clf_logits_margin, clf_labels)
    
    # 1
    prototypes = pl_module.food101_pl_head.get_prototypes()
    feature_dim = pl_module.food101_pl_head.n_hidden // 2

    if pl_module.hparams.config['missing_type'][phase] == "both":
        selected_prototypes = prototypes
        if pl_module.food101_pl_head.zerout:
            delta = selected_prototypes.clone().detach()
            delta[0].zero_()
            delta[1, :feature_dim].zero_()
            delta[2, feature_dim:].zero_()

    elif pl_module.hparams.config['missing_type'][phase] == "text":
        selected_prototypes = prototypes[:2]
        if pl_module.food101_pl_head.zerout:
            delta = selected_prototypes.clone().detach()
            delta[0].zero_()
            delta[1, :feature_dim].zero_()

    elif pl_module.hparams.config['missing_type'][phase] == "image":
        selected_prototypes = prototypes[[0, 2]]
        if pl_module.food101_pl_head.zerout:
            delta = selected_prototypes.clone().detach()
            delta[0].zero_()
            delta[1, feature_dim:].zero_()

    else:
        raise NotImplementedError
    # prototypes = F.normalize(selected_prototypes.permute(2, 0, 1), dim=-1)
    
    if pl_module.food101_pl_head.zerout:
        selected_prototypes = selected_prototypes + delta
    prototypes = selected_prototypes.permute(2, 0, 1)

    prototypes_contra_loss = pl_module.hparams.config['contrast_coef'] * SupConLoss(
        contrast_mode=pl_module.hparams.config['contrast_mode'], temperature=pl_module.hparams.config['contrast_temp'], base_temperature=pl_module.hparams.config['contrast_temp_base'])(
            prototypes,
            torch.arange(pl_module.hparams.config["food101_class_num"], dtype=torch.long, device=clf_feats.device)
            )
    overall_loss = clf_loss + prototypes_contra_loss
    
    # if pl_module.hparams.config['missing_type']['train'] == 'both':
    #     unbind_pros = torch.unbind(prototypes, dim=1)
    #     prototypes_seperate_loss = (nn.CosineSimilarity()(unbind_pros[1], unbind_pros[2])).mean() * pl_module.hparams.config['seperate_coef']
    #     overall_loss += prototypes_seperate_loss

    ret = {
        "food101_labels": clf_labels,
        "food101_logits": clf_logits,
        "food101_loss": overall_loss,
    }

    loss = getattr(pl_module, f"{phase}_food101_loss")(ret["food101_loss"])
    acc = getattr(pl_module, f"{phase}_food101_accuracy")(
        ret["food101_logits"], ret["food101_labels"]
    )
    
    pl_module.log(f"food101/{phase}/loss", loss)
    pl_module.log(f"food101/{phase}/clf_loss", clf_loss)
    pl_module.log(f"food101/{phase}/contra_loss", prototypes_contra_loss)
    # if pl_module.hparams.config['missing_type']['train'] == 'both':
    #     pl_module.log(f"food101/{phase}/seperate_loss", prototypes_seperate_loss)
    
    return ret


def compute_hatememes_plt3(pl_module, batch):
    phase = "train" if pl_module.training else "val"
    if phase == "train":
        infer = pl_module.infer(batch)
    else:
        infer = pl_module.infer(batch)

    clf_feats = pl_module.hatememes_proj(infer["cls_feats"])
    clf_labels = torch.tensor(batch["label"]).to(pl_module.device).long()
    
    m = torch.tensor(batch["missing_type"], device=pl_module.device).long()
    clf_logits = pl_module.hatememes_pl_head(clf_feats, m)
    
    clf_logits_margin = torch.where(
        m.reshape(-1, 1) == 0, pl_module.hatememes_arc_head(clf_logits, clf_labels),
        torch.where(m.reshape(-1, 1) == 1, pl_module.hatememes_arc_head_tm(clf_logits, clf_labels), pl_module.hatememes_arc_head_im(clf_logits, clf_labels))
     )
    clf_loss = F.cross_entropy(clf_logits_margin, clf_labels)
    
    # 1
    prototypes = pl_module.hatememes_pl_head.get_prototypes()
    if pl_module.hparams.config['missing_type'][phase] == "both":
        selected_prototypes = prototypes
    elif pl_module.hparams.config['missing_type'][phase] == "text":
        selected_prototypes = prototypes[:2]
    elif pl_module.hparams.config['missing_type'][phase] == "image":
        selected_prototypes = prototypes[[0, 2]]
    else:
        raise NotImplementedError
    # prototypes = F.normalize(selected_prototypes.permute(2, 0, 1), dim=-1)
    prototypes = selected_prototypes.permute(2, 0, 1)
    
    prototypes_contra_loss = pl_module.hparams.config['contrast_coef'] * SupConLoss(
        contrast_mode=pl_module.hparams.config['contrast_mode'], temperature=pl_module.hparams.config['contrast_temp'], base_temperature=pl_module.hparams.config['contrast_temp_base'])(
            prototypes,
            torch.arange(pl_module.hparams.config["hatememes_class_num"], dtype=torch.long, device=clf_feats.device)
            )
    overall_loss = clf_loss + prototypes_contra_loss

    ret = {
        "hatememes_labels": clf_labels,
        "hatememes_logits": clf_logits,
        "hatememes_loss": overall_loss,
    }

    loss = getattr(pl_module, f"{phase}_hatememes_loss")(ret["hatememes_loss"])
    acc = getattr(pl_module, f"{phase}_hatememes_accuracy")(
        ret["hatememes_logits"], ret["hatememes_labels"]
    )
    auroc = getattr(pl_module, f"{phase}_hatememes_AUROC")(
        ret["hatememes_logits"], ret["hatememes_labels"]
    )
    
    pl_module.log(f"hatememes/{phase}/loss", loss)
    pl_module.log(f"hatememes/{phase}/clf_loss", clf_loss)
    pl_module.log(f"hatememes/{phase}/contra_loss", prototypes_contra_loss)
    
    return ret


class ArcMarginProductForMultilabel(nn.Module):
    """
    https://github.com/egcode/pytorch-losses/blob/master/mnist-visualize-arcface-loss.ipynb
    """
    def __init__(self, s=64.0, m=0.50, easy_margin=False):
        super(ArcMarginProductForMultilabel, self).__init__()
        # self.in_features = in_features
        # self.out_features = out_features
        
        self.s = s
        self.m = m
        
        # self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        # nn.init.xavier_uniform_(self.weight)
        # self.device = device

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, logits, label):
        # self.cos_m = torch.tensor(self.cos_m, dtype=logits.dtype)
        # self.sin_m = torch.tensor(self.sin_m, dtype=logits.dtype)
        # self.th = torch.tensor(self.th, dtype=logits.dtype)
        # self.mm = torch.tensor(self.mm, dtype=logits.dtype)

        # self.cos_m, self.sin_m, self.th, self.mm = self.cos_m.to(logits.dtype), self.sin_m.to(logits.dtype), self.th.to(logits.dtype), self.mm.to(logits.dtype)
        
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = logits.float()
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        # one_hot = torch.zeros(cosine.size(), device=logits.device)
        # one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        one_hot = label
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output
    
    
def compute_mmimdb_plt3(pl_module, batch):
    phase = "train" if pl_module.training else "val"
    if phase == "train":
        infer = pl_module.infer(batch)
    else:
        infer = pl_module.infer(batch)

    clf_feats = pl_module.mmimdb_proj(infer["cls_feats"])
    clf_labels = torch.tensor(batch["label"]).to(pl_module.device).float()
    
    m = torch.tensor(batch["missing_type"], device=pl_module.device).long()
    clf_logits = pl_module.mmimdb_pl_head(clf_feats, m)
    
    clf_logits_margin = torch.where(
        m.reshape(-1, 1) == 0, pl_module.mmimdb_arc_head(clf_logits, clf_labels),
        torch.where(m.reshape(-1, 1) == 1, pl_module.mmimdb_arc_head_tm(clf_logits, clf_labels), pl_module.mmimdb_arc_head_im(clf_logits, clf_labels))
    )
    clf_loss = F.binary_cross_entropy_with_logits(clf_logits_margin, clf_labels)

    # 1
    prototypes = pl_module.mmimdb_pl_head.get_prototypes()
    if pl_module.hparams.config['missing_type'][phase] == "both":
        selected_prototypes = prototypes
    elif pl_module.hparams.config['missing_type'][phase] == "text":
        selected_prototypes = prototypes[:2]
    elif pl_module.hparams.config['missing_type'][phase] == "image":
        selected_prototypes = prototypes[[0, 2]]
    else:
        raise NotImplementedError
    # prototypes = F.normalize(selected_prototypes.permute(2, 0, 1), dim=-1)
    prototypes = selected_prototypes.permute(2, 0, 1)
                             
    prototypes_contra_loss = SupConLoss(
        contrast_mode=pl_module.hparams.config['contrast_mode'],
        temperature=pl_module.hparams.config['contrast_temp'],
        base_temperature=pl_module.hparams.config['contrast_temp_base'])(
            prototypes,
            torch.arange(pl_module.hparams.config["mmimdb_class_num"],
            dtype=torch.long, device=clf_feats.device)
    ) * pl_module.hparams.config['contrast_coef']
    
    overall_loss = clf_loss + prototypes_contra_loss
    
    ret = {
        "mmimdb_labels": clf_labels,
        "mmimdb_logits": clf_logits,
        "mmimdb_loss": overall_loss,
    }

    loss = getattr(pl_module, f"{phase}_mmimdb_loss")(ret["mmimdb_loss"])
    f1_scores = getattr(pl_module, f"{phase}_mmimdb_F1_scores")(
        ret["mmimdb_logits"], ret["mmimdb_labels"]
    )
    
    pl_module.log(f"mmimdb/{phase}/loss", loss)
    pl_module.log(f"mmimdb/{phase}/clf_loss", clf_loss)
    pl_module.log(f"mmimdb/{phase}/contra_loss", prototypes_contra_loss)
    
    return ret


class DecoupledType4(nn.Module):
    """
    simple decouple missing and non-missing class wise prototypes
    """
    
    def __init__(self, n_prototypes, n_hidden):
        super().__init__()
        
        self.n_prototypes = n_prototypes
        self.n_hidden = n_hidden
        
        w_mi_c = torch.empty(n_hidden, n_prototypes)
        nn.init.xavier_normal_(w_mi_c)
        self.w_mi_c = nn.Parameter(w_mi_c)
        
        w_mi_tm = torch.empty(n_hidden, n_prototypes)
        nn.init.xavier_normal_(w_mi_tm)
        self.w_mi_tm = nn.Parameter(w_mi_tm)

        w_mt_c = torch.empty(n_hidden, n_prototypes) # text modality prototype when image is complete
        nn.init.xavier_normal_(w_mt_c)
        self.w_mt_c = nn.Parameter(w_mt_c)
        
        w_mt_im = torch.empty(n_hidden, n_prototypes) # text modality prototype when image is missing
        nn.init.xavier_normal_(w_mt_im)
        self.w_mt_im = nn.Parameter(w_mt_im)

    def get_prototypes(self):
        return F.normalize(self.w_mi_c, dim=0), F.normalize(self.w_mi_tm, dim=0), F.normalize(self.w_mt_c, dim=0), F.normalize(self.w_mt_im, dim=0)
        
    def forward(self, x, missing_type):
        
        w_mi_c, w_mi_tm, w_mt_c, w_mt_im = self.get_prototypes()
        ws = torch.stack([
            torch.cat([w_mi_c, w_mt_c], dim=0),                     # complete
            torch.cat([w_mi_tm, torch.zeros_like(w_mt_c)], dim=0),  # text missing
            torch.cat([torch.zeros_like(w_mi_c), w_mt_im], dim=0),  # image missing
        ], dim=0)
        w_select = ws[missing_type]
        
        h = torch.cat([F.normalize(x[:, :self.n_hidden], dim=1), F.normalize(x[:, self.n_hidden:], dim=1)], dim=1)
        
        logits = torch.matmul(h.unsqueeze(1), w_select).squeeze(1)  # bs x n_prototypes
        logits = torch.where(missing_type.reshape(-1, 1)==0, logits / 2, logits)  # complete modality, averaging logits
        
        return logits


def compute_hatememes_plt42(pl_module, batch):
    phase = "train" if pl_module.training else "val"
    if phase == "train":
        infer = pl_module.infer(batch)
    else:
        infer = pl_module.infer(batch)

    clf_feats = pl_module.hatememes_proj(infer["cls_feats"])
    clf_labels = torch.tensor(batch["label"]).to(pl_module.device).long()
    
    m = torch.tensor(batch["missing_type"], device=pl_module.device).long()
    clf_logits = pl_module.hatememes_pl_head(clf_feats, m)
    
    clf_logits_margin = torch.where(
        m.reshape(-1, 1) == 0, pl_module.hatememes_arc_head(clf_logits, clf_labels),
        torch.where(m.reshape(-1, 1) == 1, pl_module.hatememes_arc_head_tm(clf_logits, clf_labels), pl_module.hatememes_arc_head_im(clf_logits, clf_labels))
     )
    clf_loss = F.cross_entropy(clf_logits_margin, clf_labels)
    
    # 2
    prototypes = pl_module.hatememes_pl_head.get_prototypes()
    prototypes_by_types = torch.stack(prototypes, dim=0)

    supconloss = SupConLoss(
        contrast_mode=pl_module.hparams.config['contrast_mode'],
        temperature=pl_module.hparams.config['contrast_temp'],
        base_temperature=pl_module.hparams.config['contrast_temp_base'])

    if pl_module.hparams.config['missing_type'][phase] == "both":
        # selected_prototypes = prototypes_by_types
        selected_prototypes = prototypes_by_types.permute(2, 0, 1)
        prototypes_contra_loss = \
        supconloss(
            selected_prototypes[:, [0, 1]],
            torch.arange(pl_module.hparams.config["hatememes_class_num"], dtype=torch.long, device=clf_feats.device),
        ) * pl_module.hparams.config['contrast_coef_mintra'] \
        + supconloss(
            selected_prototypes[:, [2, 3]],
            torch.arange(pl_module.hparams.config["hatememes_class_num"], dtype=torch.long, device=clf_feats.device),
        ) * pl_module.hparams.config['contrast_coef_mintra'] \
        + supconloss(
            selected_prototypes[:, [0, 2]],
            torch.arange(pl_module.hparams.config["hatememes_class_num"], dtype=torch.long, device=clf_feats.device),
        ) * pl_module.hparams.config['contrast_coef_minter_pintra'] \
        + supconloss(
            selected_prototypes[:, [1, 3]],
            torch.arange(pl_module.hparams.config["hatememes_class_num"], dtype=torch.long, device=clf_feats.device),
        ) * pl_module.hparams.config['contrast_coef_minter_pinter'] \
        + supconloss(
            selected_prototypes[:, [0, 3]],
            torch.arange(pl_module.hparams.config["hatememes_class_num"], dtype=torch.long, device=clf_feats.device),
        ) * pl_module.hparams.config['contrast_coef_minter_pinter'] \
        + supconloss(
            selected_prototypes[:, [1, 2]],
            torch.arange(pl_module.hparams.config["hatememes_class_num"], dtype=torch.long, device=clf_feats.device),
        ) * pl_module.hparams.config['contrast_coef_minter_pinter'] 

    elif pl_module.hparams.config['missing_type'][phase] == "text":
        # selected_prototypes = prototypes_by_types[[0, 1, 2]]
        selected_prototypes = prototypes_by_types[[0, 1, 2]].permute(2, 0, 1)
        prototypes_contra_loss = \
        supconloss(
            selected_prototypes[:, [0, 1]],
            torch.arange(pl_module.hparams.config["hatememes_class_num"], dtype=torch.long, device=clf_feats.device),
        ) * pl_module.hparams.config['contrast_coef_mintra'] \
        + supconloss(
            selected_prototypes[:, [0, 2]],
            torch.arange(pl_module.hparams.config["hatememes_class_num"], dtype=torch.long, device=clf_feats.device),
        ) * pl_module.hparams.config['contrast_coef_minter_pintra'] \
        + supconloss(
            selected_prototypes[:, [1, 2]],
            torch.arange(pl_module.hparams.config["hatememes_class_num"], dtype=torch.long, device=clf_feats.device),
        ) * pl_module.hparams.config['contrast_coef_minter_pinter']

    elif pl_module.hparams.config['missing_type'][phase] == "image":
        # selected_prototypes = prototypes_by_types[[0, 2, 3]]
        selected_prototypes = prototypes_by_types[[0, 2, 3]].permute(2, 0, 1)
        prototypes_contra_loss = \
        supconloss(
            selected_prototypes[:, [0, 1]],
            torch.arange(pl_module.hparams.config["hatememes_class_num"], dtype=torch.long, device=clf_feats.device),
        ) * pl_module.hparams.config['contrast_coef_minter_pintra'] + \
        supconloss(
            selected_prototypes[:, [0, 2]],
            torch.arange(pl_module.hparams.config["hatememes_class_num"], dtype=torch.long, device=clf_feats.device),
        ) * pl_module.hparams.config['contrast_coef_minter_pinter'] + \
        supconloss(
            selected_prototypes[:, [1, 2]],
            torch.arange(pl_module.hparams.config["hatememes_class_num"], dtype=torch.long, device=clf_feats.device),
        ) * pl_module.hparams.config['contrast_coef_mintra']
    else:
        raise NotImplementedError
    
    # selected_prototypes = selected_prototypes.permute(2, 0, 1)
    # prototypes_contra_loss = SupConLoss(
    #     contrast_mode=pl_module.hparams.config['contrast_mode'],
    #     temperature=pl_module.hparams.config['contrast_temp'],
    #     base_temperature=pl_module.hparams.config['contrast_temp_base'])(
    #         selected_prototypes,
    #         torch.arange(pl_module.hparams.config["hatememes_class_num"], dtype=torch.long, device=clf_feats.device)
    #     ) * pl_module.hparams.config['contrast_coef']
    
    # overall loss
    overall_loss = clf_loss + prototypes_contra_loss

    ret = {
        "hatememes_labels": clf_labels,
        "hatememes_logits": clf_logits,
        "hatememes_loss": overall_loss,
    }

    loss = getattr(pl_module, f"{phase}_hatememes_loss")(ret["hatememes_loss"])
    acc = getattr(pl_module, f"{phase}_hatememes_accuracy")(
        ret["hatememes_logits"], ret["hatememes_labels"]
    )
    auroc = getattr(pl_module, f"{phase}_hatememes_AUROC")(
        ret["hatememes_logits"], ret["hatememes_labels"]
    )
    
    pl_module.log(f"hatememes/{phase}/loss", loss)
    pl_module.log(f"hatememes/{phase}/clf_loss", clf_loss)
    pl_module.log(f"hatememes/{phase}/contra_loss", prototypes_contra_loss)
    
    return ret


def compute_hatememes_plt41(pl_module, batch):
    phase = "train" if pl_module.training else "val"
    if phase == "train":
        infer = pl_module.infer(batch)
    else:
        infer = pl_module.infer(batch)

    clf_feats = pl_module.hatememes_proj(infer["cls_feats"])
    clf_labels = torch.tensor(batch["label"]).to(pl_module.device).long()
    
    m = torch.tensor(batch["missing_type"], device=pl_module.device).long()
    clf_logits = pl_module.hatememes_pl_head(clf_feats, m)
    
    clf_logits_margin = torch.where(
        m.reshape(-1, 1) == 0, pl_module.hatememes_arc_head(clf_logits, clf_labels),
        torch.where(m.reshape(-1, 1) == 1, pl_module.hatememes_arc_head_tm(clf_logits, clf_labels), pl_module.hatememes_arc_head_im(clf_logits, clf_labels))
     )
    clf_loss = F.cross_entropy(clf_logits_margin, clf_labels)
    
    # 1
    prototypes = pl_module.hatememes_pl_head.get_prototypes()
    prototypes_by_cases = torch.stack([
            torch.cat([prototypes[0], prototypes[2]], dim=0),                     # complete
            torch.cat([prototypes[1], torch.zeros_like(prototypes[2])], dim=0),  # text missing
            torch.cat([torch.zeros_like(prototypes[0]), prototypes[3]], dim=0),  # image missing
    ], dim=0)
    if pl_module.hparams.config['missing_type'][phase] == "both":
        selected_prototypes = prototypes_by_cases
    elif pl_module.hparams.config['missing_type'][phase] == "text":
        selected_prototypes = prototypes_by_cases[[0, 1]]
    elif pl_module.hparams.config['missing_type'][phase] == "image":
        selected_prototypes = prototypes_by_cases[[0, 2]]
    else:
        raise NotImplementedError
    selected_prototypes = selected_prototypes.permute(2, 0, 1)
    
    prototypes_contra_loss = SupConLoss(
        contrast_mode=pl_module.hparams.config['contrast_mode'],
        temperature=pl_module.hparams.config['contrast_temp'],
        base_temperature=pl_module.hparams.config['contrast_temp_base'])(
            selected_prototypes,
            torch.arange(pl_module.hparams.config["hatememes_class_num"], dtype=torch.long, device=clf_feats.device)
        ) * pl_module.hparams.config['contrast_coef']
    
    # overall loss
    overall_loss = clf_loss + prototypes_contra_loss

    ret = {
        "hatememes_labels": clf_labels,
        "hatememes_logits": clf_logits,
        "hatememes_loss": overall_loss,
    }

    loss = getattr(pl_module, f"{phase}_hatememes_loss")(ret["hatememes_loss"])
    acc = getattr(pl_module, f"{phase}_hatememes_accuracy")(
        ret["hatememes_logits"], ret["hatememes_labels"]
    )
    auroc = getattr(pl_module, f"{phase}_hatememes_AUROC")(
        ret["hatememes_logits"], ret["hatememes_labels"]
    )
    
    pl_module.log(f"hatememes/{phase}/loss", loss)
    pl_module.log(f"hatememes/{phase}/clf_loss", clf_loss)
    pl_module.log(f"hatememes/{phase}/contra_loss", prototypes_contra_loss)
    
    return ret


def compute_mmimdb_plt42(pl_module, batch):
    phase = "train" if pl_module.training else "val"
    if phase == "train":
        infer = pl_module.infer(batch)
    else:
        infer = pl_module.infer(batch)

    clf_feats = pl_module.mmimdb_proj(infer["cls_feats"])
    clf_labels = torch.tensor(batch["label"]).to(pl_module.device).float()
    
    m = torch.tensor(batch["missing_type"], device=pl_module.device).long()
    clf_logits = pl_module.mmimdb_pl_head(clf_feats, m)
    
    clf_logits_margin = torch.where(
        m.reshape(-1, 1) == 0, pl_module.mmimdb_arc_head(clf_logits, clf_labels),
        torch.where(m.reshape(-1, 1) == 1, pl_module.mmimdb_arc_head_tm(clf_logits, clf_labels), pl_module.mmimdb_arc_head_im(clf_logits, clf_labels))
    )
    clf_loss = F.binary_cross_entropy_with_logits(clf_logits_margin, clf_labels)

    # 2
    prototypes = pl_module.mmimdb_pl_head.get_prototypes()
    prototypes_by_types = torch.stack(prototypes, dim=0)
    if pl_module.hparams.config['missing_type'][phase] == "both":
        selected_prototypes = prototypes_by_types = torch.stack(prototypes, dim=0)
    elif pl_module.hparams.config['missing_type'][phase] == "text":
        selected_prototypes = prototypes_by_types[[0, 1, 2]]
    elif pl_module.hparams.config['missing_type'][phase] == "image":
        selected_prototypes = prototypes_by_types[[0, 2, 3]]
    else:
        raise NotImplementedError
    selected_prototypes = selected_prototypes.permute(2, 0, 1)

    prototypes_contra_loss = SupConLoss(
        contrast_mode=pl_module.hparams.config['contrast_mode'],
        temperature=pl_module.hparams.config['contrast_temp'],
        base_temperature=pl_module.hparams.config['contrast_temp_base'])(
            selected_prototypes,
            torch.arange(pl_module.hparams.config["mmimdb_class_num"], dtype=torch.long, device=clf_feats.device)
        ) * pl_module.hparams.config['contrast_coef']
        
    # overall loss
    overall_loss = clf_loss + prototypes_contra_loss
    
    ret = {
        "mmimdb_labels": clf_labels,
        "mmimdb_logits": clf_logits,
        "mmimdb_loss": overall_loss,
    }

    loss = getattr(pl_module, f"{phase}_mmimdb_loss")(ret["mmimdb_loss"])
    f1_scores = getattr(pl_module, f"{phase}_mmimdb_F1_scores")(
        ret["mmimdb_logits"], ret["mmimdb_labels"]
    )
    
    pl_module.log(f"mmimdb/{phase}/loss", loss)
    pl_module.log(f"mmimdb/{phase}/clf_loss", clf_loss)
    pl_module.log(f"mmimdb/{phase}/contra_loss", prototypes_contra_loss)
    
    return ret


def compute_food101_plt42(pl_module, batch):
    phase = "train" if pl_module.training else "val"
    if phase == "train":
        infer = pl_module.infer(batch)
    else:
        infer = pl_module.infer(batch)

    clf_feats = pl_module.food101_proj(infer["cls_feats"])
    clf_labels = torch.tensor(batch["label"]).to(pl_module.device).long()
    
    m = torch.tensor(batch["missing_type"], device=pl_module.device).long()
    clf_logits = pl_module.food101_pl_head(clf_feats, m)
    
    clf_logits_margin = torch.where(
        m.reshape(-1, 1) == 0, pl_module.food101_arc_head(clf_logits, clf_labels),
        torch.where(m.reshape(-1, 1) == 1, pl_module.food101_arc_head_tm(clf_logits, clf_labels), pl_module.food101_arc_head_im(clf_logits, clf_labels))
     )
    clf_loss = F.cross_entropy(clf_logits_margin, clf_labels)
    
    # 2
    prototypes = pl_module.food101_pl_head.get_prototypes()
    prototypes_by_types = torch.stack(prototypes, dim=0)

    # if pl_module.hparams.config['missing_type'][phase] == "both":
    #     selected_prototypes = prototypes_by_types
    # elif pl_module.hparams.config['missing_type'][phase] == "text":
    #     selected_prototypes = prototypes_by_types[[0, 1, 2]]
    # elif pl_module.hparams.config['missing_type'][phase] == "image":
    #     selected_prototypes = prototypes_by_types[[0, 2, 3]]
    # else:
    #     raise NotImplementedError
    # selected_prototypes = selected_prototypes.permute(2, 0, 1)
    
    # prototypes_contra_loss = SupConLoss(
    #     contrast_mode=pl_module.hparams.config['contrast_mode'],
    #     temperature=pl_module.hparams.config['contrast_temp'],
    #     base_temperature=pl_module.hparams.config['contrast_temp_base'])(
    #         selected_prototypes,
    #         torch.arange(pl_module.hparams.config["food101_class_num"], dtype=torch.long, device=clf_feats.device)
    #     ) * pl_module.hparams.config['contrast_coef']

    supconloss = SupConLoss(
        contrast_mode=pl_module.hparams.config['contrast_mode'],
        temperature=pl_module.hparams.config['contrast_temp'],
        base_temperature=pl_module.hparams.config['contrast_temp_base'])

    if pl_module.hparams.config['missing_type'][phase] == "both":
        selected_prototypes = prototypes_by_types.permute(2, 0, 1)
        prototypes_contra_loss = \
        supconloss(
            selected_prototypes[:, [0, 1]],
            torch.arange(pl_module.hparams.config["food101_class_num"], dtype=torch.long, device=clf_feats.device),
        ) * pl_module.hparams.config['contrast_coef_mintra'] \
        + supconloss(
            selected_prototypes[:, [2, 3]],
            torch.arange(pl_module.hparams.config["food101_class_num"], dtype=torch.long, device=clf_feats.device),
        ) * pl_module.hparams.config['contrast_coef_mintra'] \
        + supconloss(
            selected_prototypes[:, [0, 2]],
            torch.arange(pl_module.hparams.config["food101_class_num"], dtype=torch.long, device=clf_feats.device),
        ) * pl_module.hparams.config['contrast_coef_minter_pintra'] \
        + supconloss(
            selected_prototypes[:, [1, 3]],
            torch.arange(pl_module.hparams.config["food101_class_num"], dtype=torch.long, device=clf_feats.device),
        ) * pl_module.hparams.config['contrast_coef_minter_pinter'] \
        + supconloss(
            selected_prototypes[:, [0, 3]],
            torch.arange(pl_module.hparams.config["food101_class_num"], dtype=torch.long, device=clf_feats.device),
        ) * pl_module.hparams.config['contrast_coef_minter_pinter'] \
        + supconloss(
            selected_prototypes[:, [1, 2]],
            torch.arange(pl_module.hparams.config["food101_class_num"], dtype=torch.long, device=clf_feats.device),
        ) * pl_module.hparams.config['contrast_coef_minter_pinter'] 
        
    elif pl_module.hparams.config['missing_type'][phase] == "text":
        selected_prototypes = prototypes_by_types[[0, 1, 2]].permute(2, 0, 1)
        prototypes_contra_loss = \
        supconloss(
            selected_prototypes[:, [0, 1]],
            torch.arange(pl_module.hparams.config["food101_class_num"], dtype=torch.long, device=clf_feats.device),
        ) * pl_module.hparams.config['contrast_coef_mintra'] \
        + supconloss(
            selected_prototypes[:, [0, 2]],
            torch.arange(pl_module.hparams.config["food101_class_num"], dtype=torch.long, device=clf_feats.device),
        ) * pl_module.hparams.config['contrast_coef_minter_pintra'] \
        + supconloss(
            selected_prototypes[:, [1, 2]],
            torch.arange(pl_module.hparams.config["food101_class_num"], dtype=torch.long, device=clf_feats.device),
        ) * pl_module.hparams.config['contrast_coef_minter_pinter']

    elif pl_module.hparams.config['missing_type'][phase] == "image":
        selected_prototypes = prototypes_by_types[[0, 2, 3]].permute(2, 0, 1)
        prototypes_contra_loss = \
        supconloss(
            selected_prototypes[:, [1, 2]],
            torch.arange(pl_module.hparams.config["food101_class_num"], dtype=torch.long, device=clf_feats.device),
        ) * pl_module.hparams.config['contrast_coef_mintra'] + \
        supconloss(
            selected_prototypes[:, [0, 1]],
            torch.arange(pl_module.hparams.config["food101_class_num"], dtype=torch.long, device=clf_feats.device),
        ) * pl_module.hparams.config['contrast_coef_minter_pintra'] + \
        supconloss(
            selected_prototypes[:, [0, 2]],
            torch.arange(pl_module.hparams.config["food101_class_num"], dtype=torch.long, device=clf_feats.device),
        ) * pl_module.hparams.config['contrast_coef_minter_pinter']
    
    # overall loss
    overall_loss = clf_loss + prototypes_contra_loss

    ret = {
        "food101_labels": clf_labels,
        "food101_logits": clf_logits,
        "food101_loss": overall_loss,
    }

    loss = getattr(pl_module, f"{phase}_food101_loss")(ret["food101_loss"])
    acc = getattr(pl_module, f"{phase}_food101_accuracy")(
        ret["food101_logits"], ret["food101_labels"]
    )
    
    pl_module.log(f"food101/{phase}/loss", loss)
    pl_module.log(f"food101/{phase}/clf_loss", clf_loss)
    pl_module.log(f"food101/{phase}/contra_loss", prototypes_contra_loss)
    
    return ret


class DecoupledType41(nn.Module):
    """
    NO missing aware, using entropy
    """
    
    def __init__(self, n_prototypes, n_hidden):
        super().__init__()
        
        self.n_prototypes = n_prototypes
        self.n_hidden = n_hidden
        
        w_mi_c = torch.empty(n_hidden, n_prototypes)
        nn.init.xavier_normal_(w_mi_c)
        self.w_mi_c = nn.Parameter(w_mi_c)
        
        w_mi_tm = torch.empty(n_hidden, n_prototypes)
        nn.init.xavier_normal_(w_mi_tm)
        self.w_mi_tm = nn.Parameter(w_mi_tm)

        w_mt_c = torch.empty(n_hidden, n_prototypes) # text modality prototype when image is complete
        nn.init.xavier_normal_(w_mt_c)
        self.w_mt_c = nn.Parameter(w_mt_c)
        
        w_mt_im = torch.empty(n_hidden, n_prototypes) # text modality prototype when image is missing
        nn.init.xavier_normal_(w_mt_im)
        self.w_mt_im = nn.Parameter(w_mt_im)

    def get_prototypes(self):
        return F.normalize(self.w_mi_c, dim=0), F.normalize(self.w_mi_tm, dim=0), F.normalize(self.w_mt_c, dim=0), F.normalize(self.w_mt_im, dim=0)
        
    def forward(self, x, missing_type):
        
        w_mi_c, w_mi_tm, w_mt_c, w_mt_im = self.get_prototypes()
        ws = torch.stack([
            torch.cat([w_mi_c, w_mt_c], dim=0),                     # complete
            torch.cat([w_mi_tm, torch.zeros_like(w_mt_c)], dim=0),  # text missing
            torch.cat([torch.zeros_like(w_mi_c), w_mt_im], dim=0),  # image missing
        ], dim=0)
        # w_select = ws[missing_type]
        
        h = torch.cat([F.normalize(x[:, :self.n_hidden], dim=1), F.normalize(x[:, self.n_hidden:], dim=1)], dim=1)
        
        # logits = torch.matmul(h.unsqueeze(1), w_select).squeeze(1)  # bs x n_prototypes
        logits = torch.einsum('bh,bchk->bck', h, ws.unsqueeze(0).repeat(h.shape[0], 1, 1, 1))

        m = torch.tensor([0, 1, 2], dtype=torch.long, device=logits.device)
        logits = torch.where(m.reshape(1, -1, 1).repeat(h.shape[0], 1, 1)==0, logits / 2, logits)  # complete modality, averaging logits
        
        # idx = logits.argmax(dim=1)
        # logits = logits.gather(dim=1, index=idx.unsqueeze(1)).squeeze(1)
        
        # Non MA
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * probs.log()).sum(dim=-1)
        _, prompt_idx = entropy.min(dim=-1)
        logits = logits[torch.arange(logits.shape[0]), prompt_idx]

        return logits


class MissingAwareFC(nn.Module):

    def __init__(self, hidden_size, cls_num):
        super().__init__()
        self.fc_c  = nn.Linear(hidden_size, cls_num)
        self.fc_c.apply(objectives.init_weights)

        self.fc_tm = nn.Linear(hidden_size, cls_num)
        self.fc_tm.apply(objectives.init_weights)

        self.fc_im = nn.Linear(hidden_size, cls_num)
        self.fc_im.apply(objectives.init_weights)

        self.fc = nn.ModuleList([self.fc_c, self.fc_tm, self.fc_im])

    def forward(self, x, missing_type):
        idx = torch.arange(x.size(0)).to(x.device)

        logits_chunk = []
        idx_chunk = []

        for i in range(3):  # 0 for complete, 1 and 2 for text and image missing
            select_cond = missing_type == i
            x_curr = x[select_cond]
            logits_curr = self.fc[i](x_curr)
            idx_curr = idx[select_cond]

            logits_chunk.append(logits_curr)
            idx_chunk.append(idx_curr)

        logits_ = torch.cat(logits_chunk, dim=0)
        idx_ = torch.cat(idx_chunk, dim=0)
        
        reorder_idx = torch.argsort(idx_)
        logits = logits_[reorder_idx]
        
        return logits
    

def compute_mmimdb_ma(pl_module, batch):
    phase = "train" if pl_module.training else "val"
    if phase == "train":
        infer = pl_module.infer(batch)
    else:
        infer = pl_module.infer(batch)
    
    m = torch.tensor(batch["missing_type"], device=pl_module.device).long()

    imgcls_logits = pl_module.mmimdb_classifier_ma(infer["cls_feats"], m)
    imgcls_labels = batch["label"]
    imgcls_labels = torch.tensor(imgcls_labels).to(pl_module.device).float()
    imgcls_loss = F.binary_cross_entropy_with_logits(imgcls_logits, imgcls_labels)

    ret = {
        "mmimdb_loss": imgcls_loss,
        "mmimdb_logits": imgcls_logits,
        "mmimdb_labels": imgcls_labels,
    }

    loss = getattr(pl_module, f"{phase}_mmimdb_loss")(ret["mmimdb_loss"])
    
    f1_scores = getattr(pl_module, f"{phase}_mmimdb_F1_scores")(
        ret["mmimdb_logits"], ret["mmimdb_labels"]
    )
    pl_module.log(f"mmimdb/{phase}/loss", loss)

    return ret


def compute_hatememes_ma(pl_module, batch):
    phase = "train" if pl_module.training else "val"
    if phase == "train":
        infer = pl_module.infer(batch)
    else:
        infer = pl_module.infer(batch)

    m = torch.tensor(batch["missing_type"], device=pl_module.device).long()

    imgcls_logits = pl_module.hatememes_classifier_ma(infer["cls_feats"], m)

    imgcls_labels = batch["label"]
#     imgcls_labels = torch.tensor(imgcls_labels).to(pl_module.device).float().view(-1,1)
#     imgcls_loss = F.binary_cross_entropy_with_logits(imgcls_logits, imgcls_labels)
    imgcls_labels = torch.tensor(imgcls_labels).to(pl_module.device).long()
    imgcls_loss = F.cross_entropy(imgcls_logits, imgcls_labels)

    ret = {
        "hatememes_loss": imgcls_loss,
        "hatememes_logits": imgcls_logits,
        "hatememes_labels": imgcls_labels,
    }

    loss = getattr(pl_module, f"{phase}_hatememes_loss")(ret["hatememes_loss"])
    acc = getattr(pl_module, f"{phase}_hatememes_accuracy")(
        ret["hatememes_logits"], ret["hatememes_labels"]
    )
    auroc = getattr(pl_module, f"{phase}_hatememes_AUROC")(
        ret["hatememes_logits"], ret["hatememes_labels"]
    )    
    pl_module.log(f"hatememes/{phase}/loss", loss)

    return ret


def compute_food101_ma(pl_module, batch):
    phase = "train" if pl_module.training else "val"
    if phase == "train":
        infer = pl_module.infer(batch)
    else:
        infer = pl_module.infer(batch)

    m = torch.tensor(batch["missing_type"], device=pl_module.device).long()

    imgcls_logits = pl_module.food101_classifier_ma(infer["cls_feats"], m)

    imgcls_labels = batch["label"]
    imgcls_labels = torch.tensor(imgcls_labels).to(pl_module.device).long()
    imgcls_loss = F.cross_entropy(imgcls_logits, imgcls_labels)

    ret = {
        "food101_loss": imgcls_loss,
        "food101_logits": imgcls_logits,
        "food101_labels": imgcls_labels,
    }

    loss = getattr(pl_module, f"{phase}_food101_loss")(ret["food101_loss"])
    acc = getattr(pl_module, f"{phase}_food101_accuracy")(
        ret["food101_logits"], ret["food101_labels"]
    )
    pl_module.log(f"food101/{phase}/loss", loss)

    return ret