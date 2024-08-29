import os
import torch
import torch.nn as nn
import random
from itertools import product
from copy import deepcopy
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque

from . import utils
from .model import *
from .colors import *

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        self.Transition = namedtuple('Transition', ('state', 'action', 'next', 'reward'))

    def push(self, *args):
        self.memory.append(self.Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Player:
    def __init__(self, type, style, model, criterion, optimizer, tau, batch_size, gamma, memory):
        self.type = type
        self.style = Style(**style)
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.tau = tau
        self.batch_size = batch_size
        self.gamma = gamma
        self.memory = ReplayMemory(memory)
        self.color = utils.get_color(type)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_net = None
        self.target_net = None
        self.path = utils.get_path(dir=('models', f'{self.type}'), name=f'{self.style.name}.pth')
        self.state = []
        self.next = []
        self.space = []
        self.action = None
        self.reward = 0
        self.L = 0
        self.R = 0
    
    @property
    def features(self):
        return len(self.space) + 2*len(self.state)

    def load(self, data):
        self.update('current', data)

        for counter, (block, color) in enumerate(product(self.state, COLORS)):
            action = Action(block, color)
            action.id = counter
            self.space.append(action)

        self.policy_net = globals()[self.model](input=self.features, output=len(self.space)).to(self.device)

        if os.path.exists(self.path):
            self.policy_net.load_state_dict(torch.load(self.path, weights_only=True))

        self.target_net = globals()[self.model](input=self.features, output=len(self.space)).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        criterion_name = self.criterion['name']
        criterion_params = self.criterion.get('params', {})
        self.criterion = getattr(nn, criterion_name)(**criterion_params)

        optimizer_name = self.optimizer['name']
        optimizer_params = self.optimizer.get('params', {})
        self.optimizer = getattr(optim, optimizer_name)(self.policy_net.parameters(), **optimizer_params)

    def update(self, type, data=None):
        if type == 'current':
            self.state = deepcopy(data)

            for action in self.space:
                action.block = self.state[action.block.id]

        elif type == 'next':
            self.next = deepcopy(data)

        elif type == 'net':
            tnsd = self.target_net.state_dict()
            pnsd = self.policy_net.state_dict()

            for key in pnsd:
                tnsd[key] = pnsd[key]*self.tau + tnsd[key]*(1-self.tau)

            self.target_net.load_state_dict(tnsd)

        else:
            raise ValueError('Invalid update type.')

    def explore(self):
        self.action = random.choice(self.space)
        self.action.increment('Exploration')

    def exploit(self):
        with torch.no_grad():
            s = torch.tensor([[block.color.encoding for block in self.state]], dtype=torch.float32).to(self.device)
            _, id  = self.policy_net(s).max(1)

        self.action = self.space[id]
        self.action.increment('Exploitation')

    def select(self, power=4):
        torch.set_printoptions(precision=6, sci_mode=False)

        with torch.no_grad():
            s = torch.tensor([[block.color.encoding for block in self.state]], dtype=torch.float32).to(self.device)
            qvalues = self.policy_net(s)

            exp_qvalues = torch.exp(qvalues)
            norm_qvalues = exp_qvalues / exp_qvalues.sum()

            sharp_qvalues = norm_qvalues ** power
            probs = sharp_qvalues / sharp_qvalues.sum()

            id = torch.multinomial(probs, 1).item()
            print(f"Chosen action {id} has prob: {probs[0, id].item()}")

        self.action = self.space[id]
        self.action.increment('Selection')

    def expand_memory(self):
        s = torch.tensor([[block.color.encoding for block in self.state]], dtype=torch.float32).to(self.device)
        a = torch.tensor([[self.action.id]]).to(self.device)
        n = torch.tensor([[block.color.encoding for block in self.next]], dtype=torch.float32).to(self.device)
        r = torch.tensor([self.reward]).to(self.device)

        self.memory.push(s, a, n, r)

    def optimize(self):
        if len(self.memory) < self.batch_size:
            return
        
        states, actions, nexts, rewards = self.get_batch()
        qvalues = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            target = rewards + self.gamma * self.target_net(nexts).max(1).values

        loss = self.criterion(qvalues, target.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        self.L += loss.item()

    def get_batch(self):
        transitions = self.memory.sample(self.batch_size)
        batch = self.memory.Transition(*zip(*transitions))
        fields = self.memory.Transition._fields

        return tuple([torch.cat(getattr(batch, field)).to(self.device) for field in fields])
    
class Action:
    def __init__(self, block, color):
        self.block = block
        self.color = color
        self.id = None
        self.invalid = 0
        self.winner = False
        self.times = {'Exploration': 0, 'Exploitation': 0, 'Selection': 0}

    def set_invalid(self):
        self.invalid = int(not self.block.is_uncolored())

    def increment(self, phase):
        self.times[phase] += 1

    def __eq__(self, other):
        return isinstance(other, Action) and self.block == other.block
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
class Style:
    def __init__(self, name, difficulty, taste, minimalism):
        self.name = name
        self.difficulty = difficulty
        self.taste = taste
        self.minimalism = minimalism

    def get_difficulty(self, level):
        for key, value in self.difficulty.items():
            if '-' in key:
                start, end = map(int, key.split('-'))

                if start <= level <= end:
                    return value
                
            elif '+' in key:
                threshold = int(key[:-1])

                if level >= threshold:
                    return value
            else:
                if level == int(key):
                    return value
                
        return 'Error: Level is out of range'
    
    def get_taste(self, color):
        return self.taste[color.tone]
    
    def get_minimalism(self, freq):
        return freq * self.minimalism