from __future__ import absolute_import

import mxnet as mx
import numpy as np

try:
    from utils.memonger import search_plan
except:
    import sys
    sys.path.append("../utils/")
    from memonger import search_plan

vgg_type = \
{
    'A' : [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

def conv_factory(data, num_filter, kernel, stride=(1, 1), pad=(1, 1), with_bn=False):
    net = mx.sym.Convolution(data,
                             num_filter=num_filter,
                             kernel=kernel,
                             stride=stride,
                             pad=pad)
    if with_bn:
        net = mx.sym.BatchNorm(net, fix_gamma=False)
    net = mx.sym.Activation(net, act_type="relu")
    net._set_attr(mirror_stage='True')
    return net


def vgg_body_factory(structure_list):
    net = mx.sym.Variable("data")
    for item in structure_list:
        if type(item) == str:
            net = mx.sym.Pooling(net, kernel=(2, 2), stride=(2, 2), pool_type="max")
        else:
            net = conv_factory(net, num_filter=item, kernel=(3, 3), stride=(1, 1), pad=(1, 1))
    return net


def get_symbol(net_type='D'):
    net = vgg_body_factory(vgg_type[net_type])
    # group 3
    net = mx.sym.Flatten(net)
    net = mx.sym.Dropout(net, p=0.5)
    net = mx.sym.FullyConnected(net, num_hidden=4096)
    net = mx.sym.Activation(net, act_type="relu")
    net._set_attr(mirror_stage='True')
    # group 4
    net = mx.sym.Dropout(net, p=0.5)
    net = mx.sym.FullyConnected(net, num_hidden=4096)
    net = mx.sym.Activation(net, act_type="relu")
    net._set_attr(mirror_stage='True')
    # group 5
    net = mx.sym.FullyConnected(net, num_hidden=1000)
    # TODO(xxx): Test speed difference between SoftmaxActivation and SoftmaxOutput
    net = mx.sym.SoftmaxOutput(net, name="softmax")
    return net

def get_module(ctx, is_train, is_memonger, batch_size):
    sym = get_symbol()
    dshape = (batch_size, 3, 224, 224)
    if is_memonger:
        sym = search_plan(sym, data=dshape)
    mod = mx.mod.Module(symbol=sym,
                        data_names=("data",),
                        label_names=("softmax_label",),
                        context=ctx)
    if is_train:
        mod.bind(data_shapes=[("data", dshape)], for_training=True, inputs_need_grad=False)
    else:
        mod.bind(data_shapes=[("data", dshape)], for_training=False, inputs_need_grad=False)
    mod.init_params(initializer=mx.init.Xavier(magnitude=2.))
    return mod



if __name__ == "__main__":
    mod = get_module(ctx=mx.gpu(), is_train=True, is_memonger=True, batch_size=128)
