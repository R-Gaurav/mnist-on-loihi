Implementing MNIST classifiction on Loihi-2 Neuromorphic Chip -- both on
Simulation Hardware and actual Physical Hardware (on INRC) through Lava library

The SLAYER network is trained on GPU first and then the correspondng Lava
network built from it is evaluated on the Loihi simulation (which runs on CPU)
as well as on the physical hardware (on INRC cloud). The SLAYER network is a
simple `Dense-SNN` composed of two Hidden Layers (128 and 64 neurons
respectively) and one Output Layer (10 neurons). The MNIST images are encoded to
binary spikes in the Input Layer via Rate Encoding, and classes are predicted
based on maximally spiking output neurons.

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

To execute this tutorial on actual Loihi hardware (on INRC), install rest of
the above-mentioned libraries from your INRC account.

# Execution Command
First train the SLAYER network using the following:
`python train_eval_snn.py --n_tsteps=20 --epochs=20 --backend=GPU`

The above will create and save a trained `*.net` file that will be loaded and
ported to Loihi. As part of the above command's execution, you will SLAYER
training and evaluating the `Dense-SNN`, and later, the `*.net` file loaded and
evaluated on the Loihi-2 simulation hardware, i.e., on CPU.

Once trained, you can simply evaluate it again on the Loihi-2 simulation
hardware by executing the following:
`python train_eval_snn.py --n_steps=32 --num_test_imgs=25 --backend=L2Sim`

To evaluate on the actual Loihi-2 physical chip on INRC, execute the following:
`SLURM=1 BOARD=ncl-ext-og-01 python train_eval_snn.py --n_steps=32 --num_test_imgs=25 --backend=L2Hw`

# Notes:
There's no need to download the MNIST dataset, the Lava-DL library comes
furnished with it. Also, your results may vary slightly with mine in the
tutorial. Feel free to comment there on my blog if you face any issues running
my code in this repository.

---

The code in this repo is adapted from the Loihi tutorials on deploying MNIST
trained network on Loihi and the Oxford Regression task. More details at:

https://lava-nc.org
