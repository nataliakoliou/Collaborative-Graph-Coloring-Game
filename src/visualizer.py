import pygame
import random
import time
import os

pygame.init()

GRID_SIZE = 3
CELL_SIZE = 100
MARGIN = 0
BORDER_WIDTH = 8
SCREEN_WIDTH = GRID_SIZE * CELL_SIZE + 2 * BORDER_WIDTH
SCREEN_HEIGHT = GRID_SIZE * CELL_SIZE + 2 * BORDER_WIDTH

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
COLORS = [RED, GREEN, BLUE, YELLOW]

MERGED_CELLS = [((0, 0), (0, 1)), ((1, 0), (1, 1))]

def create_grid():
    grid = [[None for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

    for merge in MERGED_CELLS:
        color = random.choice(COLORS)
        for cell in merge:
            grid[cell[0]][cell[1]] = color

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if not any((i, j) in merge for merge in MERGED_CELLS):
                grid[i][j] = random.choice(COLORS)

    return grid

def draw_grid(screen, grid):
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            x = j * CELL_SIZE + BORDER_WIDTH
            y = i * CELL_SIZE + BORDER_WIDTH

            cell_color = grid[i][j]
            pygame.draw.rect(screen, cell_color, [x, y, CELL_SIZE, CELL_SIZE])
            
            draw_borders(screen, i, j)

def draw_borders(screen, row, col):
    x = col * CELL_SIZE + BORDER_WIDTH
    y = row * CELL_SIZE + BORDER_WIDTH

    borders = {
        'left':  pygame.Rect(x - BORDER_WIDTH, y, BORDER_WIDTH, CELL_SIZE),
        'right': pygame.Rect(x + CELL_SIZE, y, BORDER_WIDTH, CELL_SIZE),
        'up':    pygame.Rect(x, y - BORDER_WIDTH, CELL_SIZE, BORDER_WIDTH),
        'down':  pygame.Rect(x, y + CELL_SIZE, CELL_SIZE, BORDER_WIDTH)
    }

    adj_cell = {
        'left':  (row, col - 1),
        'right': (row, col + 1),
        'up':    (row - 1, col),
        'down':  (row + 1, col)
        }

    for direction, border_rect in borders.items():
        adj_row, adj_col = adj_cell[direction]
        
        if 0 <= adj_row < GRID_SIZE and 0 <= adj_col < GRID_SIZE:
            is_merged = any(
                (row, col) in merge and (adj_row, adj_col) in merge
                for merge in MERGED_CELLS
            )
            if not is_merged:
                pygame.draw.rect(screen, BLACK, border_rect)
        else:
            pygame.draw.rect(screen, BLACK, border_rect)


def save_image(screen, file_path):
    pygame.image.save(screen, file_path) 

def main():
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Static Grid with Merged Cells and Borders")

    grid = create_grid() 
    screen.fill(BLACK)
    draw_grid(screen, grid)  
    pygame.display.flip()

    output_dir = os.path.join(os.path.dirname(__file__), '..', 'images') 
    os.makedirs(output_dir, exist_ok=True)  

    file_path = os.path.join(output_dir, 'grid_image_with_borders.png')

    save_image(screen, file_path)
    print(f"Image saved to {file_path}")

    time.sleep(5)

    pygame.quit()

if __name__ == "__main__":
    main()
