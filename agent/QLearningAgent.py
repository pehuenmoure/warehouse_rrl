import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .model import BaselinePolicy, RelationalPolicy
from .util import QReplayMemory, QTransition


class QLearningAgent(object):
    def __init__(
        self,
        n_actions,
        gamma,
        batch_size,
        eps,
        eps_min,
        eps_decay,
        memory_capacity,
        policy,
        n_attention_blocks=2,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        channels=3
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.eps = eps
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.memory_capacity = memory_capacity
        self.device = device

        self.policy = policy(n_attention_blocks, channels=channels).to(device).double()
        self.target = policy(n_attention_blocks, channels=channels).to(device).double()
        self.update_target()
        self.target.eval()

        self.optim = optim.RMSprop(
            self.policy.parameters(), lr=2e-4, eps=0.1, weight_decay=0.99
        )
        self.memory = QReplayMemory(memory_capacity)

        self._steps = 0

    def action(self, state):
        sample = random.random()
        threshold = self.eps_min + (self.eps - self.eps_min) * math.exp(
            -1.0 * self._steps / self.eps_decay
        )
        self._steps += 1
        if sample > threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor(
                [[random.randrange(self.n_actions)]], device=self.device
            )

    def optimize(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = QTransition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device).double()
        next_state_values[non_final_mask] = (
            self.target(non_final_next_states).max(1)[0].detach()
        )
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(
            state_action_values, expected_state_action_values.unsqueeze(1)
        )

        # Optimize the model
        self.optim.zero_grad()
        loss.backward()
        for param in self.policy.parameters():
            if param.grad is None:
                continue
            param.grad.data.clamp_(-1, 1)
        self.optim.step()

    def save(self):
        pass

    def load(self):
        pass

    def update_target(self):
        self.target.load_state_dict(self.policy.state_dict())

    def push_transition(self, *args):
        self.memory.push(*args)


class RelationalQLearningAgent(QLearningAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, policy=RelationalPolicy)


class BaselineQLearningAgent(QLearningAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, policy=BaselinePolicy)