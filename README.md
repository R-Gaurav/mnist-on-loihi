Implementing MNIST/NMINST classifiction on Loihi, training on GPU and evaluation on Loihi2.
Network is a simple Dense-SNN. Focusing on MNIST as of now.

Probable reasons for the LavaSNN to not work:

0. Check that the `net` (obtained from `netx`) doesn't spike on zero input. DONE

1. Is my spike output from the Image-to-Spike class the same as the Python class
   one?
2. Check the behaviour of the LavaSNN layers upon feeding input spikes.
3. If possible, check the behaviour of the quantized network in PyTorch itself?
4. Look for quantized training options in SLAYER.

Ways ahead?
0. Quantize and Train the SNN on your own using PyTorch, then make an exactly
   similar SNN with Lava and execute on Loihi2?

---

The code in this repo is adapted from Loihi tutorials on deploying MNIST trained
network on Loihi2, Oxford Regression task.
