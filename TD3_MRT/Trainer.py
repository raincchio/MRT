import copy
import torch
import torch.nn.functional as F
from network import Actor, QF, RewardTransformation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            ptbs=None,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            mrt_interval=500,
            mrt=False
    ):
        self.ptbs = torch.as_tensor(ptbs).float().to(device)
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.qf = QF(state_dim, action_dim, len(ptbs)).to(device)
        self.qf_target = copy.deepcopy(self.qf)
        self.qf_optimizer = torch.optim.Adam(self.qf.parameters(), lr=3e-4)

        if mrt:
            self.bias_qf = QF(state_dim, action_dim, len(ptbs)).to(device)
            self.bias_qf_optimizer = torch.optim.Adam(self.bias_qf.parameters(), lr=3e-4)

        self.mrt_interval = mrt_interval
        self.mrt = mrt
        self.bias1 = None
        self.bias2 = None

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    def select_action(self, state):
        state = torch.from_numpy(state[None]).float().to(device)
        return self.actor(state).cpu().detach().numpy()[0]

    def train(self, replay_buffer, batch_size=100, q_idx=None):


        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                    torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                    self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            next_q1s, next_q2s = self.qf_target(next_state, next_action)
            min_next_qs = torch.min(next_q1s, next_q2s)

            target_qs = reward + not_done * self.discount * min_next_qs + self.ptbs

        # Get current Q estimates
        current_q1s, current_q2s = self.qf(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_q1s, target_qs) + F.mse_loss(current_q2s, target_qs)

        # Optimize the critic
        self.qf_optimizer.zero_grad()
        critic_loss.backward()
        self.qf_optimizer.step()

        if self.mrt and self.total_it % self.mrt_interval == 0:
            self.bias1 = (current_q1s - target_qs).mean(0).detach() *(1-self.total_it/1e6)
            self.bias2 = (current_q2s - target_qs).mean(0).detach()*(1-self.total_it/1e6)

        if self.mrt:
            bias_current_q1s, bias_current_q2s = self.bias_qf(state, action)
            bias_critic_loss = (F.mse_loss(bias_current_q1s, target_qs+self.bias1)
                                + F.mse_loss(bias_current_q2s, target_qs+self.bias2)
                                )
            self.bias_qf_optimizer.zero_grad()
            bias_critic_loss.backward()
            self.bias_qf_optimizer.step()

        if self.mrt:
            q_values = self.bias_qf.Q1(state, self.actor(state))
        else:
            q_values = self.qf.Q1(state, self.actor(state))
        self.actor_optimizer.zero_grad()
        actor_loss = -q_values[:, q_idx].mean()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.total_it += 1
        # Update the frozen target models
        if self.mrt:
            for param, target_param in zip(self.bias_qf.parameters(), self.qf_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.bias_qf.parameters(), self.qf.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        else:
            for param, target_param in zip(self.qf.parameters(), self.qf_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return current_q1s.mean().item(), critic_loss.item()/2

    def save(self, filename):
        torch.save(self.qf.state_dict(), filename + "_critic")
        torch.save(self.qf_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.qf.load_state_dict(torch.load(filename + "_critic"))
        self.qf_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.qf_target = copy.deepcopy(self.qf)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
