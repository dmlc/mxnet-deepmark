from __future__ import absolute_import

import sys
sys.path.insert(0, "../../mxnet/python")

import mxnet as mx
import numpy as np

try:
    from utils.memonger import search_plan
except:
    import sys
    sys.path.append("../utils/")
    from memonger import search_plan

def Conv(data, num_filter):
    conv = mx.sym.Convolution(data, num_filter=num_filter, kernel=(3,3,3), pad=(1,1,1), stride=(1,1,1))
    relu = mx.sym.Activation(conv, act_type="relu")
    return relu

def get_symbol():
    data = mx.sym.Variable("data")
    conv1 = Conv(data, 64)
    pool1 = mx.sym.Pooling(conv1, kernel=(1,2,2), stride=(1,2,2), pool_type="max")
    conv2 = Conv(pool1, 128)
    pool2 = mx.sym.Pooling(conv2, kernel=(2,2,2), stride=(2,2,2), pool_type="max")
    conv3 = Conv(pool2, 256)
    conv4 = Conv(conv3, 256)
    pool3 = mx.sym.Pooling(conv4, kernel=(2,2,2), stride=(2,2,2), pool_type="max")
    conv5 = Conv(pool3, 512)
    conv6 = Conv(conv5, 512)
    pool4 = mx.sym.Pooling(conv6, kernel=(2,2,2), stride=(2,2,2), pool_type="max")
    conv7 = Conv(pool4, 512)
    conv8 = Conv(conv7, 512)
    pool5 = mx.sym.Pooling(conv8, kernel=(2,2,2), stride=(2,2,2), pool_type="max")
    flatten = mx.sym.Flatten(pool5)
    fc1 = mx.sym.FullyConnected(flatten, num_hidden=4096)
    relu1 = mx.sym.Activation(fc1, act_type="relu")
    dp1 = mx.sym.Dropout(relu1)
    fc2 = mx.sym.FullyConnected(dp1, num_hidden=4096)
    relu2 = mx.sym.Activation(fc2, act_type="relu")
    dp2 = mx.sym.Dropout(relu2)
    fc = mx.sym.FullyConnected(dp2, num_hidden=101)
    sm = mx.sym.SoftmaxOutput(fc, name="softmax")

    return sm,  [('data', (32,3,16,112,112))], [('softmax_label', (32,))]
