import os
import argparse
import torch
import lava.lib.dl.slayer as slayer
from torch.utils.data import DataLoader

from net_utils.utils import ExpDataset
from net_utils.snns import SlayerDenseSNN, LavaDenseSNN

class TrainEvalSNN():
  def __init__(self, device, epochs, n_tsteps):
    self.model = SlayerDenseSNN().to(device)
    self.device = device
    self.epochs = epochs
    self.n_ts = n_tsteps

  def train_eval_snn(self):
    loss = slayer.loss.SpikeRate(
        # `true_rate` and `false_rate` should be between [0, 1].
        true_rate=0.9, # Keep `true_rate` high for quicker learning.
        false_rate=0.01, # Keep `false_rate` low for quicker learning.
        reduction="sum").to(self.device)
    stats = slayer.utils.LearningStats()
    optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
    assistant = slayer.utils.Assistant(
        self.model, loss, optimizer, stats,
        classifier=slayer.classifier.Rate.predict)

    for epoch in range(1, self.epochs+1):
      # Train Model.
      self.model.train()
      train_data = ExpDataset(is_train=True, n_tsteps=self.n_ts)
      train_loader = DataLoader(train_data, batch_size=100, num_workers=4)
      for inp, lbl in train_loader:
        inp, lbl = inp.to(self.device), lbl.to(self.device)
        output = assistant.train(inp, lbl)

      # Evaluate Model.
      self.model.eval()
      test_data = ExpDataset(is_train=False, n_tsteps=self.n_ts)
      test_loader = DataLoader(test_data, batch_size=100, num_workers=4)
      for inp, lbl in test_loader:
        inp, lbl = inp.to(self.device), lbl.to(self.device)
        output = assistant.test(inp, lbl)

      # Print the Stats, Save the best test-accuracy model, and Update `stats`.
      print("Epoch: {0}, Stats: {1}".format(epoch, stats))
      if stats.testing.best_accuracy:
        torch.save(self.model.state_dict(), "./trained_mnist_network.pt")
      stats.update()

    # Now load the saved dict and export the hdf5 files.
    self.model.load_state_dict(torch.load("./trained_mnist_network.pt"))
    self.model.export_hdf5("./trained_mnist_network.net")

if __name__=="__main__":
  device = "cuda" if torch.cuda.is_available() else "cpu"
  parser = argparse.ArgumentParser()
  parser.add_argument("--epochs",type=int, default=25, required=False)
  parser.add_argument("--n_tsteps", type=int, default=32, required=False)
  parser.add_argument("--backend", type=str, default="GPU", required=False)
  parser.add_argument("--num_test_imgs", type=int, default=25, required=False)

  args = parser.parse_args()

  if args.backend == "GPU":
    print("Training and Evaluating SlayerDenseSNN on GPU, and LavaDenseSNN on "
          "Loihi-2 Simulation Hardware on CPU.")
    tes = TrainEvalSNN(
        device=device, epochs=args.epochs, n_tsteps=args.n_tsteps)
    tes.train_eval_snn()
    lava_snn = LavaDenseSNN(
        "./trained_mnist_network.net",
        img_shape=784,
        n_tsteps=args.n_tsteps,
        st_img_id=0, # Start evaluating from the 1st test image.
        num_test_imgs=args.num_test_imgs,
        )
    lava_snn.infer_on_loihi_sim()

  elif args.backend == "L2Sim" or arg.backend == "L2Hw":
    try:
      assert os.path.isfile("./trained_mnist_network.net")
    except:
      sys.exit(
          "First train SlayerDenseSNN on GPU to obtain trained weights. Exit..")

    if args.backend == "L2Sim":
      print("Only evaluating the LavaDenseSNN on Loihi-2 Simulation Hardware.")
    elif args.backend == "L2Hw":
      print("Only evaluating the LavaDenseSNN on Loihi-2 Physical Hardware.")
    lava_snn = LavaDenseSNN(
        "./trained_mnist_network.net",
        img_shape=784,
        n_tsteps=args.n_tsteps,
        st_img_id=0, # Start evaluating from the frist test image.
        num_test_imgs=arg.num_test_imgs,
        )
    lava_snn.infer_on_loihi_sim(args.backend)
