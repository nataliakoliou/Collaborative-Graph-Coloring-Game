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
        self.last_k = viz['last_k']
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

                if block.color.name == neighbor.color.name:
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
        saved_grid = utils.load_pickle(name='env')

        if saved_grid is None:
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
            
            utils.save_pickle(data=self.grid, name='env')

        else:
            self.grid = saved_grid
        
            unique_ids = set(cell.id for row in self.grid for cell in row)
            self.num_blocks = len(unique_ids)

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
        intended_reveal = random.randint(self.minR, self.maxR)
        hidden_blocks = [block for block in self.state if block.is_hidden()]

        actual_reveal = min(len(hidden_blocks), intended_reveal)
        
        for block in random.sample(hidden_blocks, actual_reveal):
            block.set_color(NC)
    
    def coordinate(self, actions):
        loser = False

        if hasattr(actions, 'human') and hasattr(actions, 'robot'):
            distinct = actions.human != actions.robot
        else:
            distinct = True

        for player in ['human', 'robot']:
            if hasattr(actions, player):
                setattr(getattr(actions, player), 'delayed', int(not distinct))

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
        gain, penalty, sanction, delay = metrics

        colored_neighbors = player.action.block.filtered_neighbors(colors=COLORS)
        level = len(colored_neighbors)
        color = player.action.color
        freq = sum(block.color.name == color.name for block in self.state)

        x = player.style.get_difficulty(level=level)
        y = player.style.get_taste(color=color)
        z = player.style.get_minimalism(freq=freq)

        xyz = utils.aggregate(values=(x,y,z), weights=player.style.weights, method='mean', remove_zeros=True)

        for neighbor in player.action.block.neighbors:
            if neighbor.color.name != player.action.color.name:
                k, m = (k + 1, m)
            else:
                k, m = (k, m + 1)

        s = player.action.invalid * sanction
        d = player.action.delayed * delay
        g = k * gain
        p = m * penalty
        pr = (1 - player.action.invalid) * xyz

        if player.action.winner:
            player.reward = s + d + g + p + pr
        else:
            player.reward = d
       
        player.R += player.reward

    def visualize(self, repeat, start, end, dir):
        pygame.init() if repeat==start else None

        if repeat % self.freq == 0 or utils.is_last(current=repeat, final=end, k=self.last_k ):

            screen = pygame.Surface((self.screen_width, self.screen_height))
            screen.fill(self.screen_color.rgb)

            self.draw_state(screen)

            path = utils.get_path(dir, name=f's{repeat}.png')
            pygame.image.save(screen, path)

            if self.live:
                display = pygame.display.set_mode((self.screen_width, self.screen_height))
                pygame.display.set_caption('State @ repeat {}'.format(repeat))
                display.blit(screen, (0, 0))
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
            names = {color.name for color in colors}
            return [neighbor for neighbor in self.neighbors if neighbor.color.name in names]

    def is_hidden(self):
        return isinstance(self.color, Hidden)
    
    def is_uncolored(self):
        return isinstance(self.color, White)
    
    def __eq__(self, other):
        return isinstance(other, Block) and self.id == other.id