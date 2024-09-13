import os
import json
import numpy as np

from . import utils

def merge_payoffs(dir):
    json_files = [f for f in os.listdir(dir) if f.endswith('.json')]

    style_to_index = {}
    index = 0
    data = []

    for filename in json_files:
        human_style, robot_style = filename.replace('.json', '').split('_x_')

        if human_style not in style_to_index:
            style_to_index[human_style] = index
            index += 1

        if robot_style not in style_to_index:
            style_to_index[robot_style] = index
            index += 1

        file_path = os.path.join(dir, filename)
        with open(file_path, 'r') as file:
            payoffs = json.load(file)
            data.append((human_style, robot_style, payoffs))

    num_styles = len(style_to_index)
    payoff_matrix = np.zeros((2, num_styles, num_styles))

    for human_style, robot_style, payoffs in data:
        human_idx = style_to_index[human_style]
        robot_idx = style_to_index[robot_style]
        
        payoff_matrix[0, human_idx, robot_idx] = payoffs['human']
        payoff_matrix[1, human_idx, robot_idx] = payoffs['robot']

    return payoff_matrix

def main():
    dir = utils.get_path(dir='static', name='payoffs')
    
    payoff_matrix = merge_payoffs(dir=dir)

if __name__ == '__main__':
    main()