from ppdet.core.workspace import register
from .utils import expand_onehot, zero_loss
from paddle.nn import functional as F
import paddle


# GHMC and GHMR are based on 
# https://github.com/open-mmlab/mmdetection/blob/v2.16.0/mmdet/models/losses/ghm_loss.py

# FocalLoss is based on
# https://github.com/open-mmlab/mmdetection/blob/v2.16.0/mmdet/models/losses/focal_loss.py

@register
class GHMCLoss(paddle.nn.Layer):
    def __init__(self,
                 bins=30,
                 momentum=0,
                 use_sigmoid=True,
                 loss_weight=1.0):
        super(GHMCLoss, self).__init__()
        assert use_sigmoid == True, 'only support sigmoid loss'
        self.use_sigmoid = use_sigmoid
        self.loss_weight = loss_weight
        self.bins = bins
        self.momentum = momentum
        edges = paddle.arange(bins + 1, dtype=paddle.float32) / bins
        self.register_buffer('edges', edges, persistable=True)
        self.edges[-1] += 1e-6
        if momentum > 0:
            acc_sum = paddle.zeros([bins])
            self.register_buffer('acc_sum', acc_sum, persistable=True)
        

    def forward(self, pred, target, avg_factor):
        if pred.size == 0:
            return zero_loss(pred)
        target.stop_gradient = True
        target = expand_onehot(target, pred.shape[-1])
        edges = self.edges
        mmt = self.momentum
        weights = paddle.zeros_like(pred)
        weights.stop_gradient = True

        pred_sigm = F.sigmoid(pred)
        pred_sigm.stop_gradient = True
        g = paddle.abs(pred_sigm - target)
        tot = max(1.0, pred.size)
        n = 0
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i+1])
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] + (1 - mmt) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
                n += 1
        if n > 0:
            weights = weights / n
        loss = F.binary_cross_entropy_with_logits(
            pred, target, reduction='none')
        # tot is used instead of avg_factor
        loss = (loss * weights).sum() / tot * self.loss_weight
        return loss



@register
class GHMRLoss(paddle.nn.Layer):
    def __init__(self,
                 mu=0.02,
                 bins=10,
                 momentum=0,
                 loss_weight=1.0):
        super(GHMRLoss, self).__init__()
        self.mu = mu
        self.bins = bins
        edges = paddle.arange(bins + 1, dtype=paddle.float32) / bins
        self.register_buffer('edges', edges, persistable=True)
        self.edges[-1] = 1e3
        self.momentum = momentum
        if momentum > 0:
            acc_sum = paddle.zeros([bins])
            self.register_buffer('acc_sum', acc_sum)
        self.loss_weight = loss_weight


    def forward(self, pred, target, avg_factor):
        if pred.size == 0:
            return zero_loss(pred)
        target.stop_gradient = True
        mu = self.mu
        edges = self.edges
        mmt = self.momentum

        diff = pred - target
        loss = paddle.sqrt(diff * diff + mu * mu) - mu

        g = paddle.abs(diff / paddle.sqrt(mu * mu + diff * diff))
        g.stop_gradient = True
        weights = paddle.zeros_like(pred)
        weights.stop_gradient = True

        tot = max(1.0, pred.size)
        n = 0
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i+1])
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                n += 1
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] + (1 - mmt) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
        if n > 0:
            weights /= n
        # tot is used instead of avg_factor
        loss = (loss * weights).sum() / tot * self.loss_weight
        return loss


@register
class FocalLoss(paddle.nn.Layer):
    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 loss_weight=1.0):
        super(FocalLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        assert use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.loss_weight = loss_weight

    def forward(self, pred, target, avg_factor):
        if pred.size == 0:
            return zero_loss(pred)
        target.stop_gradient = True
        target = expand_onehot(target, pred.shape[-1])
        pred_sigm = F.sigmoid(pred)
        pt = (1 - pred_sigm) * target + pred_sigm * (1 - target)
        focal_weight = (self.alpha * target + (1 - self.alpha) * (1 - target)) * pt.pow(self.gamma)
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none') * focal_weight
        loss = loss.sum() / avg_factor
        return loss * self.loss_weight


@register
class L1Loss(paddle.nn.Layer):
    def __init__(self,
                 loss_weight=1.0):
        super(L1Loss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, pred, target, avg_factor):
        if pred.size == 0:
            return zero_loss(pred)
        target.stop_gradient = True
        loss = paddle.abs(pred - target)
        loss = loss.sum() / avg_factor
        return loss * self.loss_weight
