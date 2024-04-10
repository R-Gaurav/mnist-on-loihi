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

# Execution Command

# Notes:

There's no need to download MNIST dataset, the Lava-DL library comes furnished
with it.
---

The code in this repo is adapted from the Loihi tutorials on deploying MNIST
trained network on Loihi and the Oxford Regression task. More details at:

https://lava-nc.org
