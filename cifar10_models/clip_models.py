import torch
import torch.nn as nn
import torch.nn.functional as F
import CLIP.clip as clip

representations_dims = {'RN50': 1024,
                        'RN101': 512,
                        'RN50x4': 640,
                        'RN50x16': 768,
                        'ViT-B/32': 512,
                        'ViT-B/16': 512}

class FC(nn.Module):
    def __init__(self, backbone="ViT-B/32", output_size=10):
        super(FC, self).__init__()
        self.bb, self.preprocess = clip.load(backbone)
        in_features = representations_dims[backbone]
        self.linear = nn.Linear(in_features=in_features, out_features=output_size)

    def forward(self, x):
        with torch.no_grad():
            x = self.bb.encode_image(x).float()
        return self.linear(x)


def ClipRN50():
    return FC('RN50')


def ClipRN101():
    return FC('RN101')


def ClipRN50x4():
    return FC('RN50x4')


def ClipRN50x15():
    return FC('RN50x16')


def ClipViTB16():
    return FC('ViT-B/16')


def ClipViTB32():
    return FC('ViT-B/32')
