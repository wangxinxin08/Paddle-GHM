from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ppdet.modeling import BaseArch
from ppdet.core.workspace import register, create

import paddle


'''
Implement the classic RetiaNet.
'''


@register
class RetinaNet(BaseArch):
    __category__ = 'architecture'
    
    def __init__(self,
                 backbone,
                 neck,
                 head):
        super(RetinaNet, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        backbone = create(cfg['backbone'])
        kwargs = {'input_shape': backbone.out_shape}
        neck = create(cfg['neck'], **kwargs)
        kwargs = {'input_shape': neck.out_shape} # not used in head
        head = create(cfg['head'], **kwargs)
        return {
            'backbone': backbone,
            'neck': neck,
            'head': head}


    def _forward(self):
        body_feats = self.backbone(self.inputs)
        fpn_feats = self.neck(body_feats)
        head_outs = self.head(fpn_feats)
        if not self.training:
            im_shape = self.inputs['im_shape']
            scale_factor = self.inputs['scale_factor']
            bboxes, bbox_num = self.head.post_process(head_outs, im_shape, scale_factor)
            return bboxes, bbox_num
        return head_outs

    # return a loss dict
    def get_loss(self):
        loss = {}
        head_outs = self._forward()
        loss_retina = self.head.get_loss(head_outs, self.inputs)
        loss.update(loss_retina)
        total_loss = paddle.add_n(list(loss.values()))
        loss.update(loss=total_loss)
        return loss

    # return {bbox: bbox_pred, bbox_num: bbox_num}
    def get_pred(self):
        bbox_pred, bbox_num = self._forward()
        output = {'bbox': bbox_pred, 'bbox_num': bbox_num}
        return output
                 
