import sys
import h5py
import logging
import numpy as np
import torch
import lava.lib.dl.slayer as slayer

from lava.proc.monitor.process import Monitor
from lava.magma.core.run_configs import Loihi2SimCfg, Loihi2HwCfg
from lava.magma.core.run_conditions import RunSteps
from lava.lib.dl import netx

from net_utils.utils import (
    InpImgToSpk, PyInpImgToSpkModel, OutSpkToCls, PyOutSpkToClsModel,
    InputAdapter, PyInputAdapter, NxInputAdapter,
    OutputAdapter, PyOutputAdapter, NxOutputAdapter,
    )

# Following class is implemented via Slayer APIs.
class SlayerDenseSNN(torch.nn.Module):
  """
  Spiking network to be trained on GPU via Slayer and then evaluated on Loihi.
  It accepts spikes from host CPU and outputs spikes corresponding to classes.
  """
  def __init__(self):
    super(SlayerDenseSNN, self).__init__()

    neuron_params = {
      "threshold": 1.0,
      "current_decay": 0.10,
      "voltage_decay": 0.10,
      "requires_grad": False,
    }

    self.blocks = torch.nn.ModuleList([
      # First Hidden Layer.
      slayer.block.cuba.Dense(
        neuron_params, 784, 128, weight_norm=False, delay=False),
      # Second Hidden Layer.
      slayer.block.cuba.Dense(
        neuron_params, 128, 64, weight_norm=False, delay=False),
      # Output Layer.
      slayer.block.cuba.Dense(
        neuron_params, 64, 10, weight_norm=False, delay=False)
    ])

  def forward(self, spikes):
    """
    Do the forward pass on the input spikes.
    """
    for block in self.blocks:
      spikes = block(spikes)

    return spikes

  def export_hdf5(self, file_name):
    """
    Export the network to hdf5 format.
    """
    h = h5py.File(file_name, "w")
    layer = h.create_group("layer")
    for i, block in enumerate(self.blocks):
      block.export_hdf5(layer.create_group(f'{i}'))

# Following class is implemented via Lava Process APIs.
class LavaDenseSNN(object):
  def __init__(self, trnd_net_path, img_shape, n_tsteps, st_img_id,
               num_test_imgs):
    """
    Args:
      trnd_net_path <str>: Trained network path.
      img_shape <int>: Shape of the flattened image.
      n_tsteps <int>: Presentation time-steps of each image.
      st_img_id <int>: The index of the test-image to start testing onwards.
      num_test_imgs <int>: Number of test images to test upon.
    """
    self.trnd_net_path = trnd_net_path
    self.img_shape = img_shape
    self.n_ts = n_tsteps
    self.curr_img_id = st_img_id
    self.num_test_imgs = num_test_imgs
    # Create Process Instances.
    # -- Spike Input Process.
    self.img_to_spk = InpImgToSpk(
        img_shape=img_shape, n_tsteps=n_tsteps, curr_img_id=st_img_id)
    # -- Lava Trained SNN portable to Loihi (either H/W or Simulation).
    self.net = netx.hdf5.Network(
        net_config=trnd_net_path, # Trained network path.
        reset_interval=n_tsteps, # Presentation time-steps of each test-image.
        reset_offset=1 # Phase shift / offset time-step to reset this network.
        )
    # -- Output Classification Process.
    self.spk_to_cls = OutSpkToCls(
        n_tsteps=n_tsteps, num_test_imgs=num_test_imgs)
    # -- Input Adapter.
    self.inp_adp = InputAdapter(shape=self.net.inp.shape)
    # -- Output Adapter.
    self.out_adp = OutputAdapter(shape=self.net.out.shape)

  def get_run_config(self, backend):
    """
    Returns the run-time config corresponding to the `backend`.

    Args:
      backend <str>: Either "L2Sim" or "L2Hw" for Loihi2SimCfg or Loihi2HwCfg.
    """
    assert backend in ["L2Sim", "L2Hw"]

    if backend == "L2Sim": # Run on the Loihi-2 Simulation Hardware on CPU.
      run_config = Loihi2SimCfg(
          select_tag="fixed_pt", # To select fixed point implementation.
          exception_proc_model_map={
            InpImgToSpk: PyInpImgToSpkModel,
            OutSpkToCls: PyOutSpkToClsModel,
            InputAdapter: PyInputAdapter,
            OutputAdapter: PyOutputAdapter
            }
          )
    elif backend == "L2Hw": # Run on the Loihi-2 Physical Hardware on INRC.
      run_config = Loihi2HwCfg(
          select_sub_proc_model=True,
          exception_proc_model_map={
            InpImgToSpk: PyInpImgToSpkModel,
            OutSpkToCls: PyOutSpkToClsModel,
            InputAdapter: NxInputAdapter,
            OutputAdapter: NxOutputAdapter
            }
          )
    return run_config

  def infer_on_loihi(self, backend):
    """
    Run inference on Loihi-2 Simulation or Physical Hardware, depending on
    `backend` argument.

    Args:
      backend <str>: Either "L2Sim" or "L2Hw" for Loihi2SimCfg or Loihi2HwCfg.
    """
    # Connect Processes.
    self.img_to_spk.spk_out.connect(self.inp_adp.inp)
    self.inp_adp.out.connect(self.net.inp)
    self.net.out.connect(self.out_adp.inp)
    self.out_adp.out.connect(self.spk_to_cls.spikes_in)

    # Connect ImgToSpk Input directly to SpkToCls Output for ground truths.
    self.img_to_spk.label_out.connect(self.spk_to_cls.label_in)

    # Get RunConfig based on the `backend`.
    run_config = self.get_run_config(backend=backend)

    # E I T H E R     E X E C U T E     T H E     F O L L O W I N G     OR
    ###########################################################################
    # Execute the trained network on ALL the RUN_TSTEPS.
    #self.img_to_spk.run(
    #    condition=RunSteps(num_steps=self.n_ts*self.num_test_imgs),
    #    run_cfg=run_config
    #    )
    ###########################################################################

    # E X E C U T E     T H I S   --    C O M M E N T     O U T     O T H E R
    ###########################################################################
    # Execute the trained network on each image indvidually.
    for _ in range(self.num_test_imgs):
      self.img_to_spk.run(
        condition=RunSteps(num_steps=self.n_ts), run_cfg=run_config
      )
    ###########################################################################

    # Infer the classes.
    ground_truths = self.spk_to_cls.true_labels.get().astype(np.int32)
    predtd_clsses = self.spk_to_cls.pred_labels.get().astype(np.int32)
    # Stop the run-time AFTER obtaining all the true and pred classes.
    self.img_to_spk.stop()
    #print("Ground Truths: ", ground_truths)
    #print("Predicted Classes: ", predtd_clsses)
    print("Accuracy on Loihi {0}: ".format(
          "Simulation" if backend=="L2Sim" else "Board"),
          np.mean(np.array(ground_truths) == np.array(predtd_clsses)))
