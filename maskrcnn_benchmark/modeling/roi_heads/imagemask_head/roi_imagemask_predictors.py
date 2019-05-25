from torch import nn
from torch.nn import functional as F

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
    def forward(self, x, img_size):
        # TODO nie biore argumentu 'features', dobrze?
        # (to nie ma tak że dobrze albo niedobrze)
        # x[0] - output z FPNa największej rozdzielczości;
        # powiększamy go do wymiarów obrazka i redukujemy głębokość
        x = x[0]
        img_size = img_size[0]
        assert (( x.size()[-2] <= x.size()[-1] ) == ( img_size[0] <= img_size[1] )), (x.size(), img_size)
        # TODO chyba img_size -> targets, wówczas w inference bez interpolacji
        x = self.interp(x, size=img_size, mode='nearest')
        print("---------------------------------------------------------------------------")
        print(x.size())
        print("---------------------------------------------------------------------------")
        x = self.conv1(x)
        assert list(x.size()) == [1, self.num_cls, img_size[0], img_size[1]]
        return x


def make_roi_imagemask_predictor(cfg):
    func = registry.ROI_IMAGEMASK_PREDICTOR[
        cfg.MODEL.ROI_IMAGEMASK_HEAD.PREDICTOR
    ]
    return func(cfg)
