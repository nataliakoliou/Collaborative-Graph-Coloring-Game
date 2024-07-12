import random
from itertools import product, chain

from .colors import *
from . import utils

class Grid:
    def __init__(self, rows, cols, merge, minR=2, wR=0.2):
        self.rows = rows
        self.cols = cols
        self.merge = merge
        self.minR = minR
        self.wR = wR
        self.grid = [[Cell(i, j) for j in range(cols)] for i in range(rows)]
        self.state = []
        self.num_blocks = 0

    @property
    def maxR(self):
        extraR = self.wR * self.num_blocks
        return int(self.minR + extraR)
    
    @property
    def constraints(self):
        constraints = set()

        for block in self.state:
            for id in block.neighbors:
                neighbor = self.state[id]

                if block.color == neighbor.color:
                    pair = frozenset({block.id, neighbor.id})
                    constraints.add(pair)

        return len(constraints)

    def load(self):
        self.load_cells()
        self.load_state()
        self.load_colors()

    def load_cells(self):
        for i, j in product(range(self.rows), range(self.cols)):
            cell = self.grid[i][j]

            cell.neighbors = [self.grid[i + dr][j + dc] for dr, dc in [(0, -1), (0, 1), (-1, 0), (1, 0)]
                              if Cell(i + dr, j + dc).is_valid(self.rows, self.cols)]
            
            ids = [neighbor.id for neighbor in cell.neighbors if neighbor.id is not None]

            if random.random() < self.merge and ids:
                cell.id = random.choice(ids)
            else:
                cell.id = self.num_blocks
                self.num_blocks += 1

    def load_state(self):
        self.state = [Block(id) for id in range(self.num_blocks)]

        for cell in chain.from_iterable(self.grid):
            self.state[cell.id].cells.append(cell)

        for block in self.state:
            shared = set()

            for cell in block.cells:
                shared |= {neighbor.id for neighbor in cell.neighbors if neighbor.id != block.id}

            block.neighbors = list(shared)

    def load_colors(self):
        max_neighbors = max(len(block.neighbors) for block in self.state)

        num_colors = max_neighbors + 1
        num_encodings = num_colors + 2

        COLORS[:] = COLORS[:num_colors]

        HIDDEN.encoding = utils.encode(k=1, n=num_encodings)
        NC.encoding = utils.encode(k=2, n=num_encodings)

        for i, color in enumerate(COLORS, start=3):
            color.encoding = utils.encode(k=i, n=num_encodings)

        self.reset()

    def reset(self):
        for block in self.state:
            block.set_color(HIDDEN)

    def step(self):
        reveal = random.randint(self.minR, self.maxR)
        hidden_blocks = [block for block in self.state if block.is_hidden()]
        r = min(len(hidden_blocks), reveal)
        
        for block in random.sample(hidden_blocks, r):
            block.set_color(NC)
    
    def coordinate(self, actions):
        loser = False

        if hasattr(actions, 'human') and hasattr(actions, 'robot'):
            distinct = actions.human != actions.robot
        else:
            distinct = False

        actions = list(actions)
        random.shuffle(actions)
    
        return actions, distinct, loser

    def apply(self, action, distinct=True, loser=False):
        action.set_invalid()
        action.winner = True

        if not bool(action.invalid):

            if distinct or not loser:
                self.state[action.block.id].set_color(action.color)
                loser = True
            else:
                action.winner = False
        
        return loser
        
    def reward(self, player, metrics):
        k, m = 0, 0
        gain, penalty, sanction = metrics

        for id in player.action.block.neighbors:
            neighbor = self.state[id]

            if neighbor.color != player.action.color:
                k, m = (k + 1, m)
            else:
                k, m = (k, m + 1)

        s = player.action.invalid * sanction
        g = k * gain
        p = m * penalty

        if player.action.winner:
            player.reward = s + g + p
        else:
            player.reward = 0

class Cell:
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.id = None
        self.color = HIDDEN
        self.neighbors = []

    def set_color(self, color):
        self.color = color

    def is_valid(self, rows, cols):
        return 0 <= self.row < rows and 0 <= self.col < cols
    
    def is_hidden(self):
        return isinstance(self.color, Hidden)
    
    def is_uncolored(self):
        return isinstance(self.color, White)

class Block:
    def __init__(self, id):
        self.id = id
        self.color = HIDDEN
        self.cells = []
        self.neighbors = []

    def set_color(self, color):
        self.color = color
        
        for cell in self.cells:
            cell.set_color(color)

    def is_hidden(self):
        return isinstance(self.color, Hidden)
    
    def is_uncolored(self):
        return isinstance(self.color, White)
    
    def __eq__(self, other):
        return isinstance(other, Block) and self.id == other.id