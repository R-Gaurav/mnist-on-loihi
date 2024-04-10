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
To execute this tutorial on your machine you only need `lava-dl` and `lava-nc`
which can be installed from
[here](https://github.com/lava-nc/lava-dl?tab=readme-ov-file#installation). Do
note that these two libraries will enable only training and evaluation on
GPU/CPU.

To execute this tutorial on actual Loihi hardware (on INRC), install rest of
the above-mentioned libraries from your INRC account.

# Execution Command

# Notes:
There's no need to download MNIST dataset, the Lava-DL library comes furnished
with it.

---

The code in this repo is adapted from the Loihi tutorials on deploying MNIST
trained network on Loihi and the Oxford Regression task. More details at:

https://lava-nc.org
