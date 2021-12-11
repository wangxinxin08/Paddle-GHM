from ppdet.modeling.proposal_generator import AnchorGenerator
from ppdet.core.workspace import register

'''
It is based on ppdet's AnchorGenerator.
First calculate anchor sizes for each feature level according to 
octave_base_scale and scales_per_octave, then feed anchor sizes 
along with other args to AnchorGenerator.
'''

@register
class RetinaAnchorGenerator(AnchorGenerator):
    def __init__(self,
                 octave_base_scale=4,
                 scales_per_octave=3,
                 aspect_ratios=[0.5, 1.0, 2.0],
                 strides=[8.0, 16.0, 32.0, 64.0, 128.0],
                 variance=[1.0, 1.0, 1.0, 1.0],
                 offset=0.0):
        anchor_sizes = []
        for s in strides:
            anchor_sizes.append([
                s * octave_base_scale * 2**(i/scales_per_octave) \
                for i in range(scales_per_octave)])
        super(RetinaAnchorGenerator, self).__init__(
            anchor_sizes=anchor_sizes,
            aspect_ratios=aspect_ratios,
            strides=strides,
            variance=variance,
            offset=offset)
