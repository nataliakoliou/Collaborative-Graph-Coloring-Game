import os
import json
import numpy as np
import matplotlib.pyplot as plt
from open_spiel.python.egt import alpharank
from open_spiel.python.egt import alpharank_visualizer
from open_spiel.python.egt import utils as egt_utils

from . import utils

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
    robot_metastrats = sorted(list(robot_styles))

    payoff_matrix = np.zeros((2, len(human_metastrats), len(robot_metastrats)))

    for human_style, robot_style, payoffs in data:
        human_idx = human_metastrats.index(human_style)
        robot_idx = robot_metastrats.index(robot_style)

        payoff_matrix[0, human_idx, robot_idx] = payoffs['human']
        payoff_matrix[1, human_idx, robot_idx] = payoffs['robot']

    return payoff_matrix, (human_metastrats, robot_metastrats)

def alpha_rank(payoff_matrix, metastrats, payoffs_are_hpt_format=False):
    rhos, rho_m, pi, _, _ = alpharank.compute(payoff_matrix, m=5, alpha=1e1)
    alpharank.print_results(payoff_matrix, payoffs_are_hpt_format, rhos=rhos, rho_m=rho_m, pi=pi)

    human_metastrats, robot_metastrats = metastrats
    strat_labels = {0: human_metastrats, 1: robot_metastrats}
    egt_utils.print_rankings_table(payoff_matrix, pi, strat_labels, num_top_strats_to_print=12)

    m_network_plotter = alpharank_visualizer.NetworkPlot(payoff_matrix, rhos, rho_m, pi, strat_labels, num_top_profiles=8)
    m_network_plotter.compute_and_draw_network()

    net_path = utils.get_path(dir=('static', 'evaluation'), name='network')
    plt.savefig(net_path, dpi=600)
    plt.close()

def main():
    dir = utils.get_path(dir=('static', 'evaluation'), name='payoffs')
    
    payoff_matrix, metastrats = init_metagame(dir=dir)
    alpha_rank(payoff_matrix, metastrats)

if __name__ == '__main__':
    main()