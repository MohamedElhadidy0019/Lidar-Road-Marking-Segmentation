# -*- coding:utf-8 -*-
# author: Xinge
# @file: cylinder_spconv_3d.py

from torch import nn

REGISTERED_MODELS_CLASSES = {}


def register_model(cls, name=None):
    global REGISTERED_MODELS_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_MODELS_CLASSES, f"exist class: {REGISTERED_MODELS_CLASSES}"
    REGISTERED_MODELS_CLASSES[name] = cls
    return cls


def get_model_class(name):
    global REGISTERED_MODELS_CLASSES
    assert name in REGISTERED_MODELS_CLASSES, f"available class: {REGISTERED_MODELS_CLASSES}"
    return REGISTERED_MODELS_CLASSES[name]


@register_model
class cylinder_asym(nn.Module):
    def __init__(self,
                 cylin_model, # the mlp point wise feature extractor (cylinder_fea)
                 segmentator_spconv, # the 3d sparse encoder-decoder netword
                 sparse_shape, # the output shape
                 ):
        super().__init__()
        self.name = "cylinder_asym"

        self.cylinder_3d_generator = cylin_model # the mlp

        self.cylinder_3d_spconv_seg = segmentator_spconv # 3d conv

        self.sparse_shape = sparse_shape

    def forward(self, train_pt_fea_ten, train_vox_ten, batch_size):


        coords, features_3d = self.cylinder_3d_generator(train_pt_fea_ten, train_vox_ten)

        #the 3d sparse encoder-decoder netword takes the output of the mlp point wise feature extractor, its coordinates
        #and uses it with the cylinder voxels as stated in the paper
        spatial_features = self.cylinder_3d_spconv_seg(features_3d, coords, batch_size)

        return spatial_features
