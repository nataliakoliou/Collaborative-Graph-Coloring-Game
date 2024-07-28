import time
import random
import pygame
from itertools import product, chain

from .colors import *
from . import utils

class Grid:
    def __init__(self, rows, cols, merge, minR=2, wR=0.2, viz={}):
        self.rows = rows
        self.cols = cols
        self.merge = merge
        self.minR = minR
        self.wR = wR
        self.freq = viz['freq']
        self.cell_size = viz['cell_size']
        self.border_width = viz['border_width']
        self.screen_color = globals().get(viz['screen_color'])()
        self.live = viz['live']
        self.screen_width = self.cols * self.cell_size + 2 * self.border_width
        self.screen_height = self.rows * self.cell_size  + 2 * self.border_width
        self.grid = [[Cell(i, j) for j in range(cols)] for i in range(rows)]
        self.state = []
        self.num_blocks = 0

    @property
    def maxR(self):
        extraR = self.wR * self.num_blocks
        return int(self.minR + extraR)
    
    @property
    def conflicts(self):
        conflicts = set()

        for block in self.state:
            for neighbor in block.filtered_neighbors(colors=COLORS):

                if block.color == neighbor.color:
                    pair = frozenset({block.id, neighbor.id})
                    conflicts.add(pair)

        return conflicts
    
    @property
    def num_conflicts(self):
        return len(self.conflicts)

    def load(self):
        self.load_cells()
        self.load_state()
        self.load_colors()

    def load_cells(self):
        for i, j in product(range(self.rows), range(self.cols)):
            cell = self.get_cell(i, j)

            cell.neighbors = [self.get_cell(i + dr, j + dc) for dr, dc in [(0, -1), (0, 1), (-1, 0), (1, 0)]
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

            ids = list(shared)
            block.neighbors = [self.state[id] for id in ids]

    def get_block(self, row, col):
        cell = self.get_cell(row, col)
        block = self.state[cell.id]

        return block
    
    def get_cell(self, row, col):
        cell = self.grid[row][col]
        
        return cell

    def load_colors(self):
        num_encodings = len(ALL)

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
        k, m, x, y, z = 0, 0, 0, 0, 0
        gain, penalty, sanction, prefs = metrics

        colored_neighbors = player.action.block.filtered_neighbors(colors=COLORS)
        x = player.style.get_difficulty(level=len(colored_neighbors))

        for neighbor in player.action.block.neighbors:

            if neighbor.color != player.action.color:
                k, m = (k + 1, m)
            else:
                k, m = (k, m + 1)

        s = player.action.invalid * sanction
        g = k * gain
        p = m * penalty
        pr = (x + y + z) * prefs / 3

        if player.action.winner:
            player.reward = s + g + p + pr
        else:
            player.reward = 0

    def visualize(self, repeat, start, end, title, phase):
        pygame.init() if repeat==start else None

        if repeat % self.freq == 0:

            screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            screen.fill(self.screen_color.rgb)

            self.draw_state(screen)

            path = utils.get_path(dir=('static', f'{title}', f'{phase}', 'viz'), 
                                  name=f'state_repeat_{repeat}.png')
            pygame.image.save(screen, path)

            if self.live:
                pygame.display.set_caption("State @ repeat {}".format(repeat))
                pygame.display.flip()

        pygame.quit() if repeat==end else None

    def draw_state(self, screen):
        for block in self.state:
            
            for cell in block.cells:
                x = cell.col * self.cell_size + self.border_width
                y = cell.row * self.cell_size + self.border_width

                pygame.draw.rect(screen, cell.color.rgb, [x, y, self.cell_size, self.cell_size])

                self.draw_borders(screen, cell, x, y)

    def draw_borders(self, screen, cell, x, y):
        borders = {
            'left': pygame.Rect(
                x - self.border_width,
                y, 
                self.border_width,
                self.cell_size
            ),
            'right': pygame.Rect(
                x + self.cell_size, 
                y, 
                self.border_width, 
                self.cell_size
            ),
            'up': pygame.Rect(
                x, 
                y - self.border_width, 
                self.cell_size, 
                self.border_width
            ),
            'down': pygame.Rect(
                x,
                y + self.cell_size,
                self.cell_size,
                self.border_width
            )
        }

        for direction, border_rect in borders.items():
            adj_row, adj_col = utils.get_adjacent_pos(row=cell.row, col=cell.col, direction=direction)
            
            if 0 <= adj_row < self.rows and 0 <= adj_col < self.cols:
                adj_cell = self.get_cell(adj_row, adj_col)

                if cell.id != adj_cell.id:
                    pygame.draw.rect(screen, self.screen_color.rgb, border_rect)
            else:
                pygame.draw.rect(screen, self.screen_color.rgb, border_rect)

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

    def filtered_neighbors(self, colors=ALL):
        if colors==ALL:
            return self.neighbors
        else:
            return [neighbor for neighbor in self.neighbors if neighbor.color in colors]

    def is_hidden(self):
        return isinstance(self.color, Hidden)
    
    def is_uncolored(self):
        return isinstance(self.color, White)
    
    def __eq__(self, other):
        return isinstance(other, Block) and self.id == other.id