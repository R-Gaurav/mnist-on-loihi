import numpy as np
from typing import Tuple
from torch.utils.data import Dataset

from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.resources import CPU, Loihi2NeuroCore
from lava.utils.dataloader.mnist import MnistDataset

from lava.utils.system import Loihi2
if Loihi2.is_loihi2_available:
  from lava.proc import embedded_io as eio

class ExpDataset(Dataset):
  def __init__(self, is_train, n_tsteps, gain=1, bias=0):
    super(ExpDataset, self).__init__()
    mnist_dset = MnistDataset() # Already has train/test flattend images.
    self.n_ts = n_tsteps
    self.gain, self.bias = gain, bias # Gain and Bias for encoding.
    if is_train:
      self.images = mnist_dset.train_images
      self.labels = np.int64(mnist_dset.train_labels) # Ensure lables are long.
    else:
      self.images = mnist_dset.test_images
      self.labels = np.int64(mnist_dset.test_labels) # Ensure labels are long.


  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx): # Returns the 784D binary spikes for one image.
    img = self.images[idx]/255 # Normalize.
    v = np.zeros_like(img.shape)
    # Last dimension of `spikes` should be time-steps and dtype should be float.
    spikes = np.zeros((img.shape[0], self.n_ts), dtype=np.float32)
    for t in range(self.n_ts):
      J = self.gain*img + self.bias
      v = v + J
      mask = v > 1.0 # 1.0 is V_threshold.
      spikes[:, t] = np.int32(mask)
      v[mask] = 0 # Hard reset voltage.

    return spikes, self.labels[idx]

###############################################################################
#################### I N P U T    P R O C E S S #######################
###############################################################################

class InpImgToSpk(AbstractProcess):
  """
  Input process to convert flattened images to binary spikes.
  """
  def __init__(self, img_shape, n_tsteps, curr_img_id, v_thr=1):
    super().__init__()
    self.spk_out = OutPort(shape=(img_shape, ))
    self.label_out = OutPort(shape=(1, ))

    self.curr_img_id = Var(shape=(1, ), init=curr_img_id)
    self.n_ts = Var(shape=(1, ), init=n_tsteps)
    self.inp_img = Var(shape=(img_shape, ))
    self.ground_truth_label = Var(shape=(1, ))
    self.v = Var(shape=(img_shape, ), init=0)
    self.vth = Var(shape=(1, ), init=v_thr)

@implements(proc=InpImgToSpk, protocol=LoihiProtocol)
@requires(CPU)
class PyInpImgToSpkModel(PyLoihiProcessModel):
  """
  Python implementation for the above `InpImgToSpk` process.
  """
  spk_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, bool, precision=1)
  label_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32, precision=32)

  curr_img_id: int = LavaPyType(int, int, precision=32)
  n_ts: int = LavaPyType(int, int, precision=32)
  inp_img: np.ndarray = LavaPyType(np.ndarray, float)
  ground_truth_label: int = LavaPyType(int, int, precision=32)
  v: np.ndarray = LavaPyType(np.ndarray, float)
  vth: float = LavaPyType(float, float)

  def __init__(self, proc_params):
    super().__init__(proc_params=proc_params)
    self.mnist_dset = MnistDataset()
    self.gain = 1
    self.bias = 0

  def post_guard(self):
    """
    Guard function for post-management phase, necessary to update the next image
    index after the current image is processed.

    Note: The execution control calls `post_guard()` after `run_spk()` every
    time-step, before updating the `self.time_step` variable to next time-step.
    """
    if self.time_step % self.n_ts == 1: # n_ts steps passed, one image processed.
      return True

    return False

  def run_post_mgmt(self):
    """
    Post-management phase executed only when the above `post_guard()` returns
    True -> then, move to the next image, reset the neuron states, etc.
    """
    img = self.mnist_dset.test_images[self.curr_img_id]
    self.inp_img = img/255
    self.ground_truth_label = self.mnist_dset.test_labels[self.curr_img_id]
    self.label_out.send(np.array([self.ground_truth_label]))
    self.v = np.zeros(self.v.shape, dtype=float)
    self.curr_img_id += 1

  def run_spk(self):
    """
    Spiking phase, this is executed every simulation time-step unconditionally,
    and first in order of all the phases.
    """
    if self.time_step % self.n_ts == 1:
      self.inp_img = np.zeros(self.inp_img.shape, dtype=float)
      self.v = np.zeros(self.v.shape, dtype=float)

    J = self.gain*self.inp_img + self.bias
    self.v[:] = self.v[:] + J[:]
    mask = self.v > self.vth
    self.v[mask] = 0
    self.spk_out.send(mask)

###############################################################################
#################### O U T P U T    P R O C E S S #####################
###############################################################################

class OutSpkToCls(AbstractProcess):
  """
  Output process to collect output neuron spikes and infer predicted class.
  """
  def __init__(self, n_tsteps, num_test_imgs, n_cls_shape=(10, )):
    super().__init__()
    self.spikes_in = InPort(shape=n_cls_shape) # Receives output spikes.
    self.label_in = InPort(shape=(1, )) # Receives ground truth labels.
    self.spikes_accum = Var(shape=n_cls_shape) # Accum. spikes for prediction.
    self.n_ts = Var(shape=(1, ), init=n_tsteps) # Image presentation time.
    self.pred_labels = Var(shape=(num_test_imgs, ))
    self.true_labels = Var(shape=(num_test_imgs, ))

@implements(proc=OutSpkToCls, protocol=LoihiProtocol)
@requires(CPU)
class PyOutSpkToClsModel(PyLoihiProcessModel):
  spikes_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool, precision=1)
  label_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int, precision=32)
  spikes_accum: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=32)
  n_ts: int = LavaPyType(int, int, precision=32)
  pred_labels: np.ndarray = LavaPyType(np.ndarray, int, precision=32)
  true_labels: np.ndarray = LavaPyType(np.ndarray, int, precision=32)

  def __init__(self, proc_params):
    super().__init__(proc_params=proc_params)
    self.curr_idx = 0

  def post_guard(self):
    """
    Guard function for Post-Management phase.
    """
    if self.time_step % self.n_ts == 0:
      return True

    return False

  def run_post_mgmt(self):
    """
    Post-Management phase: executed only when the guard function above returns
    True.
    """
    true_label = self.label_in.recv()
    pred_label = np.argmax(self.spikes_accum)
    self.true_labels[self.curr_idx] = true_label[0]
    self.pred_labels[self.curr_idx] = pred_label
    self.curr_idx += 1
    self.spikes_accum = np.zeros_like(self.spikes_accum)

  def run_spk(self):
    """
    Spiking phase: executed unconditionally at every time-step, first in order
    among all the phases.
    """
    spk_in = self.spikes_in.recv()
    self.spikes_accum = self.spikes_accum + spk_in

###############################################################################
################# I N P    A D A P T E R    P R O C E S S #####################
###############################################################################

class InputAdapter(AbstractProcess):
  """
  Input Adapter Process.
  """
  def __init__(self, shape: Tuple[int, ...]):
    super().__init__(shape=shape)
    self.inp = InPort(shape=shape)
    self.out = OutPort(shape=shape)

@implements(proc=InputAdapter, protocol=LoihiProtocol)
@requires(CPU)
class PyInputAdapter(PyLoihiProcessModel):
  """
  Input adapter model for CPU, i.e., when your spike input process is on CPU and
  you plan to send the input spikes to a Loihi2 Simulation running on CPU.
  """
  inp: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
  out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

  def run_spk(self):
    self.out.send(self.inp.recv())

@implements(proc=InputAdapter, protocol=LoihiProtocol)
@requires(Loihi2NeuroCore)
class NxInputAdapter(AbstractSubProcessModel):
  """
  Input adapter model for Loihi-2, i.e., your spikes are generated on CPU and
  you plan to send them to Loihi-2 neuro-cores.
  """
  def __init__(self, proc: AbstractProcess):
    self.inp: PyInPort = LavaPyType(np.ndarray, np.int32)
    self.out: PyOutPort = LavaPyType(np.ndarray, np.int32)
    shape = proc.proc_params.get("shape")
    self.adapter = eio.spike.PyToNxAdapter(shape=shape)
    proc.inp.connect(self.adapter.inp)
    self.adapter.out.connect(proc.out)

###############################################################################
################# O U T    A D A P T E R    P R O C E S S #####################
###############################################################################

class OutputAdapter(AbstractProcess):
  """
  Output adapater process.
  """
  def __init__(self, shape: Tuple[int, ...]):
    super().__init__(shape=shape)
    self.inp = InPort(shape=shape)
    self.out = OutPort(shape=shape)

@implements(proc=OutputAdapter, protocol=LoihiProtocol)
@requires(CPU)
class PyOutputAdapter(PyLoihiProcessModel):
  """
  Output adapter model for CPU, i.e., when your SNN is running on Loihi2
  Simulation on CPU, and you plan to accept the output spikes on CPU itself.
  """
  inp: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
  out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

  def run_spk(self):
    self.out.send(self.inp.recv())

@implements(proc=OutputAdapter, protocol=LoihiProtocol)
@requires(Loihi2NeuroCore)
class NxOutputAdapter(AbstractSubProcessModel):
  """
  Output adapter mode for Loihi-2, i.e., your spikes are generated on Loihi-2
  neuro-cores and you plan to send them to CPU.
  """
  def __init__(self, proc:AbstractProcess):
    self.inp: PyInPort = LavaPyType(np.ndarray, np.int32)
    self.out: PyOutPort = LavaPyType(np.ndarray, np.int32)

    shape = proc.proc_params.get("shape")
    self.adapter = eio.spike.NxToPyAdapter(shape=shape)
    proc.inp.connect(self.adapter.inp)
    self.adapter.out.connect(proc.out)
