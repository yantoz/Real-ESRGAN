dependencies = ['torch']

import torch.hub

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

class _esrgan():

    def __init__(self, scale, model):
        self._scale = scale
        self._model = model

    @property
    def scale(self):
        return self._scale

    def __call__(self, *args):
        return self._model(*args)

def ESRGAN(network='RealESRGAN_x4plus', progress=True, map_location=None):

    if network == 'RealESRGAN_x4plus':
        params = [4, RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)]
        url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
    elif network == 'RealESRNet_x4plus':
        params =  [4, RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)]
        url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth'
    elif network == 'RealESRGAN_x4plus_anime_6B':
        params = [4, RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6,  num_grow_ch=32, scale=4)]
        url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth'
    elif network == 'RealESRGAN_x2plus':
        params = [2, RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)]
        url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth'
    elif network == 'realesr-animevideov3':
        params = [4, SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')]
        url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth'
    elif network == 'realesr-general-wdn-x4v3':
        params = [4, SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')]
        url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth'
    elif network == 'realesr-general-x4v3':
        params = [4, SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')]
        url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
    else:
        raise Exception("Unknown model: {}".format(network))

    checkpoint = torch.hub.load_state_dict_from_url(url, progress=progress, map_location=map_location) 
    keys = list(checkpoint.keys())
    if len(keys) == 1:
        checkpoint = checkpoint[keys[0]]

    model = params[1]
    model.load_state_dict(checkpoint, strict=True)
    model.eval()

    if map_location:
        model = model.to(map_location)

    model = _esrgan(params[0], model)
    return model

