import torch
import torch.nn as nn
import torch.nn.functional as F

def AvgL1Norm(x, eps=1e-8):
    return x/x.abs().mean(-1,keepdim=True).clamp(min=eps)


class RewardTransformation(nn.Module):
	def __init__(self):
		super(RewardTransformation, self).__init__()


	def forward(self, state, action, zsa, zs):
		sa = torch.cat([state, action], 1)
		embeddings = torch.cat([zsa, zs], 1)

		q1 = AvgL1Norm(self.q01(sa))
		q1 = torch.cat([q1, embeddings], 1)
		q1 = self.activ(self.q1(q1))
		q1 = self.activ(self.q2(q1))
		q1 = self.q3(q1)

		return q1