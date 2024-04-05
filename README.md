Implementing MNIST classifiction on Loihi-1 or Loihi-2 Neuromorphic Simulation
Hardware through Lava library

The SLAYER network is trained on GPU and is evaluated on the Loihi simulation
which runs on CPUs. The SLAYER network is a simple Dense-SNN composed of two
Hidden Layers (128 and 64 neurons respectively) and one Output Layer. The MNIST
images are encoded to binary spikes in the Input Layer via Rate Encoding.

---

The code in this repo is adapted from the Loihi tutorials on deploying MNIST
trained network on Loihi and the Oxford Regression task. More details at:

https://lava-nc.org
