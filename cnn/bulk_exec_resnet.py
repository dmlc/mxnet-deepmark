import sys
import os
os.environ["MXNET_EXEC_PREFER_BULK_EXEC"] = sys.argv[1]

import numpy as np
import mxnet as mx
import resnet
import time

batch_size = 1
dshape = (batch_size, 3, 224, 224)

layers = [3, 24, 36, 3]
factor = 1
cfg = [int(z * factor) for z in layers]
nmodule = sum(cfg)
net = resnet.get_symbol(cfg)
print("Test resnet with %d modules.. check if bulk message appear" % nmodule)

def test_speed(net):
    dev = mx.gpu(0)
    dshape = (batch_size, 3, 224, 224)
    lshape = (batch_size)
    num_epoch = 100
    tmp_data = np.random.uniform(-1, 1, dshape).astype("float32")

    train_iter = mx.io.NDArrayIter(data=tmp_data,  batch_size=batch_size, shuffle=False, last_batch_handle='pad')

    model = net.simple_bind(ctx=dev, grad_req="null", data=dshape)
    print("Temp space: ", model.debug_str().split('\n')[-3])

    arg_names = net.list_arguments()
    arg_map = dict(zip(arg_names, model.arg_arrays))

    param_blocks = [(i, arg_map[arg_names[i]]) for i in range(len(arg_names))]

    input_ndarray = arg_map["data"]
    grad = mx.nd.zeros((batch_size, 1000), ctx=mx.gpu())
    param_len = len(param_blocks)

    for i in range(param_len):
        param_blocks[i][1][:] = mx.rnd.uniform(-0.01, 0.01, param_blocks[i][1].shape)

    train_iter.reset()
    dbatch = train_iter.next()
    dbatch.data[0].copyto(input_ndarray)
    model.forward(is_train=False)

    # block all async all
    mx.nd.waitall()

    tic = time.time()

    def test_forward(model, epoch):
        tic = time.time()
        for i in range(epoch):
            model.forward(is_train=False)
            # Note: This command will force thread engine block, which hurts performance a lot
            # Remove it will bring parallelism bias
            model.outputs[0].wait_to_read()
        toc = time.time()
        return (toc - tic) / epoch
    fwd = test_forward(model, num_epoch)
    print("Avg forward per batch: ", fwd)

test_speed(net)
