import paddle
from ppdet.core.workspace import register
from ppdet.modeling.proposal_generator.target import label_box


'''
It uses ppdet's label_bbox as backend.
It does assignment for a batch of images.
'''

@register
class RetinaBBoxAssigner(object):
    def __init__(self, fg_thresh=0.5, bg_thresh=0.4, allow_low_quality=True):
        self.fg_thresh = fg_thresh
        self.bg_thresh = bg_thresh
        self.allow_low_quality = allow_low_quality

    def __call__(self, anchors, meta):
        if isinstance(anchors, (list, tuple)):
            anchors = paddle.concat(anchors)
        matches_list, match_labels_list = [], []
        for gt_bbox, gt_class in zip(meta['gt_bbox'], meta['gt_class']):
            matches, match_labels = label_box(
                anchors,
                gt_bbox,
                positive_overlap=self.fg_thresh,
                negative_overlap=self.bg_thresh,
                allow_low_quality=self.allow_low_quality,
                ignore_thresh=None,
                is_crowd=None,
                assign_on_cpu=False)
            matches_list.append(matches)
            match_labels_list.append(match_labels)
        return matches_list, match_labels_list
