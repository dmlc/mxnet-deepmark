# Image and Video benchmarks

## How to use:

First install MXNet. Once done, one can test the performance on Alexnet with
batch size 32 on GPU 0 by

```bash
python benchmark.py --network alexnet --batch-size 32 --gpus 0
```

It will output the time (in ms) for running a single iteration (forward +
backward + parameter update). So the images processed per second can be obtained
by `bath_size / time_in_ms * 1000`.

To run on GPU 0, 1, 2, and 3, we can

```bash
python benchmark.py --network alexnet --batch-size 32 --gpus 0,1,2,3
```

Note that the `batch-size` is the total batch size for all GPUs, in the above
example, each GPU only gets batch size 8. We often need to increase this number
for better performance.

The best performance is often the largest batch size which can fit into GPU
memory.

For different networks such as `vgg`, `inceptionv3`, and `resnet`, we can set it
by the `--network` option.
