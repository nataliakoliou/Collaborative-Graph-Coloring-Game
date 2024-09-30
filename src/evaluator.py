import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from open_spiel.python.egt import alpharank
from open_spiel.python.egt import alpharank_visualizer
from open_spiel.python.egt import utils as egt_utils

from . import utils

np.set_printoptions(suppress=True, precision=18)

config = utils.load_yaml(path=utils.get_path(dir=(os.path.dirname(__file__), '..'), name='config.yaml'))

level = config['track']['logger']
path = utils.get_path(dir=('static', 'evaluation'), name='loggings.pth')
logger = utils.get_logger(level=level, path=path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f'Device is {device}')

def min_max_scaling(matrix, new_min, new_max):
    flattened = matrix.flatten()

    min_val = np.min(flattened)
    max_val = np.max(flattened)

    scaled_matrix = np.empty_like(matrix, dtype=float)
    
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            for k in range(matrix.shape[2]):
                scaled_matrix[i, j, k] = new_min + (matrix[i, j, k] - min_val) * (new_max - new_min) / (max_val - min_val)
    
    return scaled_matrix

def init_metagame(dir):
    json_files = [f for f in os.listdir(dir) if f.endswith('.json')]

    human_styles = set()
    robot_styles = set()
    data = []

    for filename in json_files:
        human_style, robot_style = filename.replace('.json', '').split('_x_')

        human_styles.add(human_style)
        robot_styles.add(robot_style)

        file_path = os.path.join(dir, filename)
        with open(file_path, 'r') as file:
            payoffs = json.load(file)
            data.append((human_style, robot_style, payoffs))

    human_metastrats = sorted(list(human_styles))
    logger.info(f'Human meta-strategies: {human_metastrats}')
    
    robot_metastrats = sorted(list(robot_styles))
    logger.info(f'Robot meta-strategies: {robot_metastrats}')

    payoff_matrix = np.zeros((2, len(human_metastrats), len(robot_metastrats)))

    for human_style, robot_style, payoffs in data:
        human_idx = human_metastrats.index(human_style)
        robot_idx = robot_metastrats.index(robot_style)

        payoff_matrix[0, human_idx, robot_idx] = payoffs['human']
        payoff_matrix[1, human_idx, robot_idx] = payoffs['robot']

    scaled_payoff_matrix = min_max_scaling(matrix=payoff_matrix, new_min=1, new_max=2)

    return scaled_payoff_matrix, (human_metastrats, robot_metastrats)

def draw_payoff_matrix(payoff_matrix, human_metastrats, robot_metastrats):
    num_rows = len(human_metastrats)
    num_cols = len(robot_metastrats)

    _, ax = plt.subplots(figsize=(num_cols, num_rows))

    ax.set_axis_off()
    table_data = [[f"({payoff_matrix[0, i, j]:.2f}, {payoff_matrix[1, i, j]:.2f})"
                   for j in range(num_cols)] for i in range(num_rows)]

    table = ax.table(cellText=table_data, rowLabels=human_metastrats, colLabels=robot_metastrats, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    png_path = utils.get_path(dir=('static', 'evaluation'), name='matrix.png')

    plt.savefig(png_path, dpi=600, bbox_inches='tight')
    plt.close()

    txt_path = utils.get_path(dir=('static', 'evaluation'), name='payoffs.txt')
    
    with open(txt_path, 'w') as txt_file:
        for i, human_strat in enumerate(human_metastrats):
            for j, robot_strat in enumerate(robot_metastrats):
                entry = f'({human_strat}, {robot_strat}): ({payoff_matrix[0, i, j]:.2f}, {payoff_matrix[1, i, j]:.2f})\n'
                txt_file.write(entry)

def alpha_rank(payoffs, strategies, alpha, population_size, num_top_strats_to_print, num_top_profiles, payoffs_are_hpt_format=False):
    rhos, rho_m, pi, _, _ = alpharank.compute(payoffs, m=population_size, alpha=alpha)
    alpharank.print_results(payoffs, payoffs_are_hpt_format, rhos=rhos, rho_m=rho_m, pi=pi)

    human_strategies, robot_strategies = strategies
    strat_labels = {0: human_strategies, 1: robot_strategies}
    egt_utils.print_rankings_table(payoffs, pi, strat_labels, num_top_strats_to_print)

    m_network_plotter = alpharank_visualizer.NetworkPlot(payoffs, rhos, rho_m, pi, strat_labels, num_top_profiles)
    m_network_plotter.compute_and_draw_network()

    plt.gcf().set_size_inches(14, 14)

    for text in plt.gca().texts:
        text.set_fontsize(10)

    net_path = utils.get_path(dir=('static', 'evaluation'), name='network.png')
    plt.savefig(net_path, dpi=600)
    plt.close()

def main():
    dir = utils.get_path(dir=('static', 'evaluation'), name='payoffs')
    
    payoff_matrix, metastrats = init_metagame(dir=dir)

    draw_payoff_matrix(payoff_matrix, *metastrats)

    alpha_rank(payoffs=payoff_matrix, 
               strategies=metastrats,
               alpha=1,
               population_size=100,
               num_top_strats_to_print=10,
               num_top_profiles=9,
               payoffs_are_hpt_format=False)

if __name__ == '__main__':
    main()