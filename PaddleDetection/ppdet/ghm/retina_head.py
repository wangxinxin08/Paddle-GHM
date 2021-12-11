from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle, math
import paddle.nn.functional as F

from ppdet.core.workspace import register
from ppdet.modeling.bbox_utils import bbox2delta, delta2bbox, clip_bbox
from paddle.nn.initializer import Normal, Constant
from paddle import ParamAttr
from .utils import batch_reshape, batch_transpose



'''
The design of interface follows ppdet's GFLHead.
'''

@register
class RetinaHead(paddle.nn.Layer):
    __inject__ = ['conv_feat', 'anchor_generator', 'loss_class',
                  'loss_bbox', 'bbox_assigner', 'nms']

    def __init__(self,
                 num_classes=80,
                 prior_prob=0.01,
                 nms_pre=1000,
                 conv_feat=None,
                 anchor_generator=None,
                 loss_class=None,
                 loss_bbox=None,
                 bbox_assigner=None,
                 nms=None):
        super(RetinaHead, self).__init__()
        self.num_classes = num_classes
        self.prior_prob = prior_prob
        self.nms_pre = nms_pre
        self.conv_feat = conv_feat
        self.anchor_generator = anchor_generator
        self.loss_class = loss_class
        self.loss_bbox  = loss_bbox 
        self.bbox_assigner = bbox_assigner
        self.nms = nms

        assert loss_class.use_sigmoid, 'only support sigmoid'
        self.cls_out_channels = num_classes

        bias_init_value = - math.log((1 - self.prior_prob) / self.prior_prob)
        self.retina_cls = self.add_sublayer(
            'retina_cls',
            paddle.nn.Conv2D(
                in_channels=conv_feat.feat_out,
                out_channels=self.cls_out_channels * anchor_generator.num_anchors,
                kernel_size=3,
                stride=1,
                padding=1,
                weight_attr=ParamAttr(initializer=Normal(mean=0.0, std=0.01)),
                bias_attr=ParamAttr(initializer=Constant(value=bias_init_value))))

        self.retina_reg = self.add_sublayer(
            'retina_reg',
            paddle.nn.Conv2D(
                in_channels=conv_feat.feat_out,
                out_channels=4 * anchor_generator.num_anchors,
                kernel_size=3,
                stride=1,
                padding=1,
                weight_attr=ParamAttr(initializer=Normal(mean=0.0, std=0.01)),
                bias_attr=ParamAttr(initializer=Constant(value=0))))


    def forward(self, fpn_feats):
        cls_logits_list = []
        bboxes_reg_list = []
        for fpn_feat in fpn_feats:
            conv_cls_feat, conv_reg_feat = self.conv_feat(fpn_feat)
            cls_logits = self.retina_cls(conv_cls_feat)
            bbox_reg = self.retina_reg(conv_reg_feat)
            cls_logits_list.append(cls_logits)
            bboxes_reg_list.append(bbox_reg)
        return (cls_logits_list, bboxes_reg_list)


    # map targets from image to feature levels
    def image_to_level(self, mask, pred, tar, num_list):
        mask = paddle.cast(mask, dtype='int32')
        mask_list = paddle.split(mask, num_list)
        tot_num = 0
        pred_list, tar_list = [], []
        for cur_mask in mask_list:
            cur_num = cur_mask.sum().item()
            if cur_num == 0:
                pred_list.append(None)
                tar_list.append(None)
            else:
                pred_list.append(pred[tot_num:tot_num+cur_num])
                tar_list.append(tar[tot_num:tot_num+cur_num])
            tot_num += cur_num
        return pred_list, tar_list

    def get_loss(self, head_outputs, meta):
        cls_logits, bboxes_reg = head_outputs
        anchors = self.anchor_generator(cls_logits)
        level_sizes = [_.shape[0] for _ in anchors]
        anchors = paddle.concat(anchors)

        # matches: include gt_inds
        # match_labels: -1, 0 or 1
        matches_list, match_labels_list = self.bbox_assigner(anchors, meta)
        cls_logits = batch_transpose(cls_logits, [0, 2, 3, 1])
        cls_logits = batch_reshape(cls_logits, [0, -1, self.cls_out_channels])
        bboxes_reg = batch_transpose(bboxes_reg, [0, 2, 3, 1])
        bboxes_reg = batch_reshape(bboxes_reg, [0, -1, 4])
        cls_logits = paddle.concat(cls_logits, axis=1)
        bboxes_reg = paddle.concat(bboxes_reg, axis=1)
        
        avg_factors = []
        cls_pred_list, cls_tar_list, reg_pred_list, reg_tar_list = [], [], [], []
        # find targets on each image
        for matches, match_labels, cls_logit, bbox_reg, gt_bbox, gt_class in zip(
                matches_list, match_labels_list, cls_logits, bboxes_reg,
                meta['gt_bbox'], meta['gt_class']):
            pos_mask = (match_labels == 1)
            neg_mask = (match_labels == 0)
            chosen_mask = paddle.logical_or(pos_mask, neg_mask)

            # add bg label to gt_class
            gt_class = gt_class.reshape([-1])
            gt_class = paddle.concat([
                gt_class,
                paddle.to_tensor([self.num_classes], dtype=gt_class.dtype, place=gt_class.place)])
            # probably there's a better way
            matches = paddle.cast(matches, dtype='float32')
            matches[neg_mask] = gt_class.size - 1
            matches = paddle.cast(matches, dtype='int64')
            
            cls_pred = cls_logit[chosen_mask]
            cls_tar  = gt_class[matches[chosen_mask]]
            reg_pred = bbox_reg[pos_mask].reshape([-1, 4])
            reg_tar  = gt_bbox[matches[pos_mask]].reshape([-1, 4])
            pos_anchors = anchors[pos_mask].reshape([-1, 4])
            # encode bbox target
            reg_tar  = bbox2delta(pos_anchors, reg_tar, weights=[1.0, 1.0, 1.0, 1.0])
            # map preds and targets to levels
            cls_pred_split, cls_tar_split = self.image_to_level(
                chosen_mask, cls_pred, cls_tar, level_sizes)
            cls_pred_list.append(cls_pred_split)
            cls_tar_list.append(cls_tar_split)
            reg_pred_split, reg_tar_split = self.image_to_level(
                pos_mask, reg_pred, reg_tar, level_sizes)
            reg_pred_list.append(reg_pred_split)
            reg_tar_list.append(reg_tar_split)
            avg_factors.append(pos_mask.sum().item())

        avg_factor = sum(avg_factors)
        cls_losses, reg_losses = [], []
        num_levels = len(level_sizes)
        # calculate loss on each level
        for lvl in range(num_levels):
            cls_pred = [_[lvl] for _ in cls_pred_list if _[lvl] is not None]
            cls_tar  = [_[lvl] for _ in cls_tar_list if _[lvl] is not None]
            if len(cls_pred) > 0:
                cls_pred = paddle.concat(cls_pred)
                cls_tar  = paddle.concat(cls_tar)
                cls_losses.append(self.loss_class(cls_pred, cls_tar, avg_factor=avg_factor))
            reg_pred = [_[lvl] for _ in reg_pred_list if _[lvl] is not None]
            reg_tar  = [_[lvl] for _ in reg_tar_list if _[lvl] is not None]
            if len(reg_pred) > 0:
                reg_pred = paddle.concat(reg_pred)
                reg_tar  = paddle.concat(reg_tar)
                reg_losses.append(self.loss_bbox(reg_pred, reg_tar, avg_factor=avg_factor))
        losses = dict(loss_cls=paddle.add_n(cls_losses), loss_reg=paddle.add_n(reg_losses))
        return losses

    # get bboxes and scores on one image
    def get_bboxes_single(self,
                          anchors,
                          cls_scores,
                          bbox_preds,
                          im_shape,
                          scale_factor,
                          rescale=True):
        assert len(cls_scores) == len(bbox_preds)
        mlvl_bboxes = []
        mlvl_scores = []
        for anchor, cls_score, bbox_pred in zip(anchors, cls_scores, bbox_preds):
            cls_score = cls_score.reshape([-1, self.cls_out_channels])
            bbox_pred = bbox_pred.reshape([-1, 4])
            if self.nms_pre is not None and cls_score.shape[0] > self.nms_pre:
                max_score = cls_score.max(axis=1)
                _, topk_inds = max_score.topk(self.nms_pre)
                bbox_pred = bbox_pred.gather(topk_inds)
                anchor    = anchor.gather(topk_inds)
                cls_score = cls_score.gather(topk_inds)
            bbox_pred = delta2bbox(bbox_pred, anchor, weights = [1.0, 1.0, 1.0, 1.0])
            bbox_pred = bbox_pred.squeeze()
            if im_shape is not None:
                bbox_pred = clip_bbox(bbox_pred, im_shape)
            mlvl_bboxes.append(bbox_pred)
            mlvl_scores.append(F.sigmoid(cls_score))
        mlvl_bboxes = paddle.concat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes = mlvl_bboxes / paddle.concat([scale_factor[::-1], scale_factor[::-1]])
        mlvl_scores = paddle.concat(mlvl_scores)
        mlvl_scores = mlvl_scores.transpose([1, 0])
        return mlvl_bboxes, mlvl_scores

    # transform network outputs to pre-nms bbox predictions on all images
    def decode(self, anchors, cls_scores, bbox_preds, im_shape, scale_factor):
        batch_bboxes = []
        batch_scores = []
        for img_id in range(cls_scores[0].shape[0]):
            num_lvls = len(cls_scores)
            cls_score_list = [cls_scores[i][img_id] for i in range(num_lvls)]
            bbox_pred_list = [bbox_preds[i][img_id] for i in range(num_lvls)]
            bboxes, scores = self.get_bboxes_single(
                anchors,
                cls_score_list,
                bbox_pred_list,
                im_shape[img_id],
                scale_factor[img_id])
            batch_bboxes.append(bboxes)
            batch_scores.append(scores)
        batch_bboxes = paddle.stack(batch_bboxes, axis=0)
        batch_scores = paddle.stack(batch_scores, axis=0)
        return batch_bboxes, batch_scores
        
    # transform network outputs to final detection results
    def post_process(self, head_outputs, im_shape, scale_factor):
        cls_scores, bbox_preds = head_outputs
        anchors = self.anchor_generator(cls_scores)
        cls_scores = batch_transpose(cls_scores, [0, 2, 3, 1])
        bbox_preds = batch_transpose(bbox_preds, [0, 2, 3, 1])
        bboxes, scores = self.decode(anchors, cls_scores, bbox_preds, im_shape, scale_factor)

        bbox_pred, bbox_num, _ = self.nms(bboxes, scores)
        return bbox_pred, bbox_num
