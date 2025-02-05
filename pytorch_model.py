import torch
import pickle

class PoseClassifier(torch.nn.Module):
  def __init__(self, weight_path):
    super().__init__()
    # weight_path: pickled weight path from original pose classifier tensorflow
    #     {"weights":[linear1, bias1, linear2, bias2, linear3, bias3]}
    # (34, 512)
    # (512,)
    # (512, 256)
    # (256,)
    # (256, 2)
    # (2,)
    loaded_data = None
    with open(weight_path, "rb") as f:
      loaded_data = pickle.load(f)
    self.weights, self.metadata = loaded_data["weights"], loaded_data["metadata"]
    self.model = torch.nn.Sequential(
        torch.nn.Linear(34, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 2),
        torch.nn.Softmax()
    )

    with torch.no_grad():
      self.model[0].weight.copy_(torch.tensor(weights[0]).T)
      self.model[0].bias.copy_(torch.tensor(weights[1]))
      self.model[2].weight.copy_(torch.tensor(weights[2]).T)
      self.model[2].bias.copy_(torch.tensor(weights[3]))
      self.model[4].weight.copy_(torch.tensor(weights[4]).T)
      self.model[4].bias.copy_(torch.tensor(weights[5]))


  def forward(self, processed_lmList):
    return self.model(processed_lmList)
