import torch
import numpy as np
from torch import nn

model_name = "TrainedModels/HarmOsc/HarmOscRandN40M20Const01Tau50TH_3eta1_2L4n0m"
model, *_ = torch.load(model_name)

L = model[0].L
d = model[0].d
n = 4

Wp = []
wp = []
bp = []
Wq = []
wq = []
bq = []

idx = 0
with torch.no_grad():
    for param in model.parameters():
        idx += 1
        
        if idx % 6 == 1:
            Wp.append(param)
        elif idx % 6 == 2:
            wp.append(param)
        elif idx % 6 == 3:
            bp.append(param)
        elif idx % 6 == 4:
            Wq.append(param)
        elif idx % 6 == 5:
            wq.append(param)
        elif idx % 6 == 0:
            bq.append(param)

