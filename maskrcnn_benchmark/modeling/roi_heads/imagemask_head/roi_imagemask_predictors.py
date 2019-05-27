from torch import nn
from torch.nn import functional as F
import torch

from maskrcnn_benchmark.modeling import registry


@registry.ROI_IMAGEMASK_PREDICTOR.register("MaskRCNNFPNImageMaskPredictor")
class MaskRCNNImageMakPredictor(nn.Module):
    def __init__(self, cfg):
        super(MaskRCNNImageMakPredictor, self).__init__()
        self.cfg = cfg
        self.in_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
        assert self.in_channels == 256
        self.num_cls = cfg.MODEL.ROI_IMAGEMASK_HEAD.NUM_CLASSES
        assert self.num_cls == 60
        self.interp = F.interpolate
        self.conv1 = nn.Conv2d(self.in_channels, self.num_cls, kernel_size=1)

    # Semantyczna segmentacja nie używa proposali z sieci RPN -
    # bierze bezpośrednio output z FPN, upsampluje do rozmiaru obrazka
    # i liczy loss.
    def forward(self, x, img_sizes):

        def including_rectangle(shapes):
            w, h = 0, 0
            for shape in shapes:
                w = max(w, shape[-2])
                h = max(h, shape[-1])
            return [w, h]

        assert False, x.shape
        x = x[0] # output z FPNa największej rozdzielczości

        new_shape = [x[0], x[1]] + including_rectangle(img_sizes) # batch size, num_features, w, h

        res = torch.zeros(new_shape)
        for i, single_feature_map in enumerate(x):
            img_size = (img_sizes[-2], img_sizes[-1])
            res[i, :, :img_sizes[i][-2], :img_sizes[i][-1]] = self.interp(single_feature_map,
                                                                          size=img_size,
                                                                          mode='nearest')

        # x = self.interp(x, size=img_sizes, mode='nearest')
        print("---------------------------------------------------------------------------")
        print(res.size())
        print("---------------------------------------------------------------------------")
        res = self.conv1(res)
        print(res.size())
        print("---------------------------------------------------------------------------")
        # assert list(x.size()) == [1, self.num_cls, img_sizes[0], img_sizes[1]]
        return res


def make_roi_imagemask_predictor(cfg):
    func = registry.ROI_IMAGEMASK_PREDICTOR[
        cfg.MODEL.ROI_IMAGEMASK_HEAD.PREDICTOR
    ]
    return func(cfg)
