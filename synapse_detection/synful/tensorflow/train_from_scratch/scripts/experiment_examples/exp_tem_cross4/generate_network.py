import json

import numpy as np
import tensorflow as tf

from funlib.learn.tensorflow import models


def mknet(parameter, name):
    learning_rate = parameter['learning_rate']
    input_shape = tuple(parameter['input_size'])
    fmap_inc_factor = parameter['fmap_inc_factor']
    downsample_factors = parameter['downsample_factors']
    fmap_num = parameter['fmap_num']
    unet_model = parameter['unet_model']
    num_heads = 2 if unet_model == 'dh_unet' else 1
    m_loss_scale = parameter['m_loss_scale']
    d_loss_scale = parameter['d_loss_scale']
    voxel_size = tuple(parameter['voxel_size']) # only needed for computing
    # field of view. No impact on the actual architecture.
    #kernel_size_down = parameter['kernel_size_down']

    assert unet_model == 'vanilla' or unet_model == 'dh_unet', \
        'unknown unetmodel {}'.format(unet_model)

    tf.reset_default_graph()

    # d, h, w
    raw = tf.placeholder(tf.float32, shape=input_shape)

    # b=1, c=1, d, h, w
    raw_batched = tf.reshape(raw, (1, 1) + input_shape)

    # b=1, c=fmap_num, d, h, w
    outputs, fov, voxel_size = models.unet(raw_batched,
                                           fmap_num, fmap_inc_factor,
                                           downsample_factors,
                                           num_heads=num_heads,
                                           voxel_size=voxel_size,
                                           #kernel_size_down=kernel_size_down
                                           )
    if num_heads == 1:
        outputs = (outputs, outputs)
    print('unet has fov in nm: ', fov)

    # b=1, c=3, d, h, w
    partner_vectors_batched, fov = models.conv_pass(
        outputs[0],
        kernel_sizes=[1],
        num_fmaps=3,
        activation=None,  # Regression
        name='partner_vector')

    # b=1, c=1, d, h, w
    syn_indicator_batched, fov = models.conv_pass(
        outputs[1],
        kernel_sizes=[1],
        num_fmaps=1,
        activation=None,
        name='syn_indicator')
    print('fov in nm: ', fov)

    # d, h, w
    output_shape = tuple(syn_indicator_batched.get_shape().as_list()[
                         2:])  # strip batch and channel dimension.
    syn_indicator_shape = output_shape

    # c=3, d, h, w
    partner_vectors_shape = (3,) + syn_indicator_shape

    # c=3, d, h, w
    pred_partner_vectors = tf.reshape(partner_vectors_batched,
                                      partner_vectors_shape)
    gt_partner_vectors = tf.placeholder(tf.float32, shape=partner_vectors_shape)
    vectors_mask = tf.placeholder(tf.float32,
                                  shape=syn_indicator_shape)  # d,h,w
    gt_mask = tf.placeholder(tf.bool, shape=syn_indicator_shape)  # d,h,w
    vectors_mask = tf.cast(vectors_mask, tf.bool)

    # d, h, w
    pred_syn_indicator = tf.reshape(syn_indicator_batched,
                                    syn_indicator_shape)  # squeeze batch dimension
    gt_syn_indicator = tf.placeholder(tf.float32, shape=syn_indicator_shape)
    indicator_weight = tf.placeholder(tf.float32, shape=syn_indicator_shape)

    partner_vectors_loss_mask = tf.losses.mean_squared_error(
        gt_partner_vectors,
        pred_partner_vectors,
        tf.reshape(
            vectors_mask,
            (1,) + syn_indicator_shape),
        reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)

    syn_indicator_loss_weighted = tf.losses.sigmoid_cross_entropy(
        gt_syn_indicator,
        pred_syn_indicator,
        indicator_weight,
        reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
    pred_syn_indicator_out = tf.sigmoid(pred_syn_indicator)  # For output.

    iteration = tf.Variable(1.0, name='training_iteration', trainable=False)
    loss = m_loss_scale * syn_indicator_loss_weighted + d_loss_scale * partner_vectors_loss_mask

    # Monitor in tensorboard.

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('loss_vectors', partner_vectors_loss_mask)
    tf.summary.scalar('loss_indicator', syn_indicator_loss_weighted)
    summary = tf.summary.merge_all()

    # l=1, d, h, w
    print("input shape : %s" % (input_shape,))
    print("output shape: %s" % (output_shape,))

    # Train Ops.
    opt = tf.train.AdamOptimizer(
        learning_rate=learning_rate,
        beta1=0.95,
        beta2=0.999,
        epsilon=1e-8
    )

    gvs_ = opt.compute_gradients(loss)
    optimizer = opt.apply_gradients(gvs_, global_step=iteration)

    tf.train.export_meta_graph(filename=name + '.meta')

    names = {
        'raw': raw.name,
        'gt_partner_vectors': gt_partner_vectors.name,
        'pred_partner_vectors': pred_partner_vectors.name,
        'gt_syn_indicator': gt_syn_indicator.name,
        'pred_syn_indicator': pred_syn_indicator.name,
        'pred_syn_indicator_out': pred_syn_indicator_out.name,
        'indicator_weight': indicator_weight.name,
        'vectors_mask': vectors_mask.name,
        'gt_mask': gt_mask.name,
        'loss': loss.name,
        'optimizer': optimizer.name,
        'summary': summary.name,
        'input_shape': input_shape,
        'output_shape': output_shape
    }

    names['outputs'] = {'pred_syn_indicator_out':
                            {"out_dims": 1, "out_dtype": "uint8"},
                        'pred_partner_vectors': {"out_dims": 3,
                                                 "out_dtype": "float32"}}
    if m_loss_scale == 0:
        names['outputs'].pop('pred_syn_indicator_out')
    if d_loss_scale == 0:
        names['outputs'].pop('pred_partner_vectors')

    with open(name + '_config.json', 'w') as f:
        json.dump(names, f)

    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print("Number of parameters:", total_parameters)
    print("Estimated size of parameters in GB:",
          float(total_parameters) * 8 / (1024 * 1024 * 1024))


if __name__ == "__main__":
    """
    
    Script to generate a tensorflow network. Needs to be run before training.
    
    This script generates a tensorflow meta file (train_net.meta ) that defines network 
    architecture and training parameters for tensorflow. It also generates a 
    config file for the gunpowder train script (train_net_config.json).
    

    Argument 1: parameter json file path.
    
    Example usage: python generate_network.py parameter.json
    """

    with open('parameter.json') as f:
        parameter = json.load(f)

    mknet(parameter, name='train_net')

    # Bigger network used for large datasets, make it as big as gpu memory allows.
    parameter['input_size'] = (860, 860, 860)
    mknet(parameter, name='test_net')
