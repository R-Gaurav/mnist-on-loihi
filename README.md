Implementing MNIST classifiction on Loihi-2 Neuromorphic Chip -- both on
Simulation Hardware and actual Physical Hardware (on [INRC](https://intel-ncl.atlassian.net/wiki/spaces/INRC/overview)) 
through Lava library; a detailed tutorial can be found [here](https://r-gaurav.github.io/2024/04/13/Lava-Tutorial-MNIST-Training-on-GPU-and-Evaluation-on-Loihi2.html).

The SLAYER network is trained on GPU first and then the correspondng Lava
network built from it is evaluated on the Loihi simulation (which runs on CPU),
as well as on the physical hardware (on INRC cloud). The SLAYER network is a
simple `Dense-SNN` composed of two Hidden Layers ($128$ and $64$ neurons
respectively) and one Output Layer ($10$ neurons). The MNIST images are encoded to
binary spikes via Rate Encoding, and classes are predicted based on maximally 
spiking output neurons. Note that you dot not need to download the MNIST images, 
it comes furnished with the Lava-DL library.

# Library Requirements:
This tutorial was evaluated on the following libraries:
```
lava-dl                   0.5.0
lava-dnf                  0.2.0
lava-loihi                0.6.0
lava-nc                   0.9.0
lava-optimization         0.4.0
lava-peripherals          0.1.0
```
To execute this tutorial on your machine, you only need `lava-dl` and `lava-nc`
which can be installed from
[here](https://github.com/lava-nc/lava-dl?tab=readme-ov-file#installation). Do
note that these two libraries will enable only training and evaluation on
GPU/CPU.

To execute this tutorial on an actual Loihi hardware (on INRC), install rest of
the above-mentioned libraries from your INRC account.

# Execution Commands
First train the SLAYER network using the following:

`python train_eval_snn.py --n_tsteps=20 --epochs=20 --backend=GPU`

The above will create and save a trained `*.net` file that will be loaded and
ported to Loihi via `netx`. As part of the above command's execution, SLAYER will
train and evaluate the `Dense-SNN` on the entire set og training and test images. 
Later, the saved `*.net` file will be loaded and evaluated on $25$ test-images on 
the Loihi-2 simulation hardware, i.e., on CPU.

After training, you can simply evaluate it again on the Loihi-2 simulation
hardware by executing the following:

`python train_eval_snn.py --n_steps=32 --num_test_imgs=25 --backend=L2Sim`

To evaluate on the actual Loihi-2 physical chip on INRC, execute the following:

`SLURM=1 BOARD=ncl-ext-og-01 python train_eval_snn.py --n_steps=32 --num_test_imgs=25 --backend=L2Hw`

# Notes:
Your results may vary slightly with mine in the tutorial. Feel free to comment there 
on my blog if you face any issues running my code in this repository. You can also find 
me on Discord at [Open Neuromorphic](https://open-neuromorphic.org/).

The code in this repo is adapted from the Loihi tutorials on deploying [MNIST
trained network on Loihi](https://lava-nc.org/lava/notebooks/end_to_end/tutorial01_mnist_digit_classification.html) 
and the Oxford Regression task: [training on GPU](https://lava-nc.org/lava-lib-dl/slayer/notebooks/oxford/train.html) 
and [inference on Loihi](https://lava-nc.org/lava-lib-dl/netx/notebooks/oxford/run.html). More details at: https://lava-nc.org
