import torch
import numpy as np
from torch import nn

# find device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))
torch.set_num_threads(1)

model_name = "TrainedModels/HarmOsc/schHarmOscRandN40M20Const01Tau50TH_3eta1_5eta2_2L4n0m"
model, *_ = torch.load(model_name)

with torch.no_grad():
    a = torch.tensor([0.5, 3.1], dtype=torch.float32, device=device).reshape((1, 1, 2))
    tau = torch.tensor([0.1], dtype=torch.float32, device=device)
    b, _ = model(a, tau)
    c, _ = model.back(b, tau)

difference = c-a
print(difference)