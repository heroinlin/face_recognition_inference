from yacs.config import CfgNode as CN

_C = CN()

_C.INPUT = CN()

_C.INPUT = CN()
_C.INPUT.HEIGHT = 112
_C.INPUT.WIDTH = 112
_C.INPUT.PIXEL_MEAN = [0.4914, 0.4822, 0.4465]
_C.INPUT.PIXEL_STD =  [0.247, 0.243, 0.261]
_C.INPUT.FORMAT = "RGB"
