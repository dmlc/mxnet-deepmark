import mxnet as mx
import math

def ConvModule(sym, num_filter, kernel, pad=(0, 0), stride=(1, 1), fix_gamma=False, with_relu=True):
    conv = mx.sym.Convolution(data=sym, kernel=kernel, stride=stride, pad=pad, num_filter=num_filter, no_bias=True)
    bn = mx.sym.BatchNorm(data=conv, fix_gamma=fix_gamma)
    act = mx.sym.Activation(data=bn, act_type="relu")  # same memory to our act, less than CuDNN one
    return act if with_relu else bn 

def ResModule(sym, base_filter, stage, layer, fix_gamma=False):
    num_f = base_filter * int(math.pow(2, stage))
    s = 1
    if stage != 0 and layer == 0:
        s = 2
    conv1 = ConvModule(sym, num_f, kernel=(1, 1), pad=(0, 0), stride=(1, 1))
    conv2 = ConvModule(conv1, num_f, kernel=(3, 3), pad=(1, 1), stride=(s, s))
    conv3 = ConvModule(conv2, num_f * 4, kernel=(1, 1), pad=(0, 0), stride=(1, 1), with_relu=False)

    if layer == 0:
        sym = ConvModule(sym, num_f * 4, kernel=(1, 1), pad=(0, 0), stride=(s, s), with_relu=False) 

    sum_sym = sym + conv3
    sum_sym = mx.symbol.Activation(data=sum_sym, act_type='relu')
    force = layer % 2 == 1
    sym._set_attr(mirror_stage='True')
    return sum_sym

def get_symbol(layers=[3, 4, 6, 3]):
    """Get a 4-stage residual net, with configurations specified as layers.

    Parameters
    ----------
    layers : list of stage configuratrion
    """
    assert(len(layers) == 4)
    base_filter = 64
    data = mx.sym.Variable(name='data')
    data = mx.sym.BatchNorm(data=data, fix_gamma=True, name='bn_data')
    conv1 = ConvModule(data, base_filter, kernel=(7, 7), pad=(3, 3), stride=(2, 2))
    mp1 = mx.sym.Pooling(data=conv1, pool_type="max", kernel=(3, 3), stride=(2, 2))
    sym = mp1
    for j in range(len(layers)):
        for i in range(layers[j]):
            sym = ResModule(sym, base_filter, j, i)

    avg = mx.symbol.Pooling(data=sym, global_pool=True, kernel=(7, 7), name="global_pool", pool_type='avg')
    flatten = mx.symbol.Flatten(data=avg, name='flatten')
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=1000, name='fc1')
    net = mx.symbol.SoftmaxOutput(data=fc1, name='softmax')
    return net, [('data', (64, 3, 224, 224))], [('softmax_label', (64,))]
