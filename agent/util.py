import os
import shutil

import numpy as np
import torchvision.transforms as transforms
from collections import namedtuple
import random

QTransition = namedtuple("QTransition", ("state", "action", "next_state", "reward"))


def generate_env_set(seeds, n, goal_lengths, num_distractors, distractor_length):
    for seed in seeds:
        env = BoxWorld(
            n,
            np.random.choice(goal_lengths),
            np.random.choice(num_distractors),
            distractor_length,
        )
        env.seed(seed)
        yield env


def boxworld_state_to_tensor(x):
    return transforms.functional.to_tensor(x).unsqueeze(0).double()


def mkdir(path, overwrite=False):
    if os.path.isdir(path) and not overwrite:
        raise FileExistsError
    elif os.path.isdir(path) and overwrite:
        shutil.rmtree(path)
    os.makedirs(path)

    import random

class QReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = QTransition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)