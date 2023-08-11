# -*- coding:utf-8 -*-
# author: Xinge
# @file: model_builder.py

from network.cylinder_spconv_3d import get_model_class
from network.segmentator_3d_asymm_spconv import Asymm_3d_spconv
from network.cylinder_fea_generator import cylinder_fea

def build(model_config):
    '''
    build model

    Args:
        model_config: dict, the dict that contains the model parameters
    Returns:
        model
    '''
    output_shape = model_config['output_shape']
    num_class = model_config['num_class']
    num_input_features = model_config['num_input_features']
    use_norm = model_config['use_norm']
    init_size = model_config['init_size']
    fea_dim = model_config['fea_dim']
    out_fea_dim = model_config['out_fea_dim']


    #? generates the whole encoder-decoder network
    cylinder_3d_spconv_seg = Asymm_3d_spconv(
        output_shape=output_shape,
        use_norm=use_norm,
        num_input_features=num_input_features,
        init_size=init_size,
        nclasses=num_class)


    #? the mlp point wise feature extractor
    cy_fea_net = cylinder_fea(grid_size=output_shape,
                              fea_dim=fea_dim,
                              out_pt_fea_dim=out_fea_dim,
                              fea_compre=num_input_features)


    #? most probably it calls this function 'cylinder_asym' from file 'cylinder_spconv_3d.py'
    model = get_model_class(model_config["model_architecture"])(
        cylin_model=cy_fea_net, #* the mlp point wise feature extractor
        segmentator_spconv=cylinder_3d_spconv_seg, #* the whole encoder-decoder network
        sparse_shape=output_shape
    )

    return model
