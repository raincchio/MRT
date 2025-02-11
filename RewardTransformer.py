import torch
import torch.nn as nn
import torch.nn.functional as F

def AvgL1Norm(x, eps=1e-8):
    return x/x.abs().mean(-1,keepdim=True).clamp(min=eps)


class RewardTransformation(nn.Module):
	def __init__(self):
		super(RewardTransformation, self).__init__()
		self.a = nn.Parameter(torch.tensor(10.0))
		self.b = nn.Parameter(torch.tensor(0.0))

	def forward(self, r):
		a = self.a.clamp(min=0.5)

		return r +self.b

	def reset(self):
		self.a = nn.Parameter(torch.tensor(10.0))
		self.b = nn.Parameter(torch.tensor(0.0))