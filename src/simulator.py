import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from . import utils
from .game import Game
from .grid import Grid
from .player import Player

config = utils.load_yaml(path=utils.get_path(dir=(os.path.dirname(__file__), '..'), name='config.yaml'))

level = config['track']['logger']
path = utils.get_path(dir=('static'), name='loggings.pth')
logger = utils.get_logger(level=level, path=path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f'Device is {device}')

def __stats__(players, top_k, steps, game):
    values, ticks, colors, names = [], [], [], []
    labels = ('Action', 'Frequency')
    func = plt.bar
    path = utils.get_path(dir=('static', 'simulation', f'{game.title}'), name='statistics.png')
    title = 'Statistics'

    for player in players:
        actions_and_freqs = [
            (f'(B{action.block.id}, {action.color.name})', action.times['Selection'] / steps)
            for action in player.space
        ]
        
        actions_and_freqs.sort(key=lambda x: x[1], reverse=True)
        top_actions_and_freqs = actions_and_freqs[:top_k]
        
        actions, freqs = zip(*top_actions_and_freqs)
        
        x_values = list(range(top_k))
        x_ticks = (x_values, list(actions))
        
        values.append((x_values, list(freqs)))
        ticks.append((x_ticks, None))
        colors.append(player.color)
        names.append(player.type)
    
    return {'values': values, 'labels': labels, 'func': func, 'path': path, 'title': title, 'colors': colors, 'ticks': ticks, 'names': names}

def __rewards__(types, rewards, colors, repeats, game):
    values = [(list(range(repeats)), rewards[type]) for type in types]
    labels = ('Step', 'Reward')
    func = plt.plot
    path = utils.get_path(dir=('static', 'simulation', f'{game.title}'), name='rewards.png')
    title = 'Reward Curve'

    return {'values': values, 'labels': labels, 'func': func, 'path': path, 'title': title, 'colors': colors, 'names': types}

def __mistakes__(mistakes, repeats, game):
    values = [(list(range(repeats)), mistakes)]
    labels = ('Repeat', 'Mistakes')
    func = plt.plot
    path = utils.get_path(dir=('static', 'simulation', f'{game.title}'), name='mistakes.png')
    title = 'Mistakes'
    colors = ['green']

    return {'values': values, 'labels': labels, 'func': func, 'path': path, 'title': title, 'colors': colors}

def simulate(game, repeats, visualize, top_k):
    env = game.env
    players = utils.filterout(input=game.players)
    types = [player.type for player in players]
    colors = [player.color for player in players]

    rewards = {type: [] for type in types}
    mistakes = []
    steps = 0

    game.load()

    pbar = tqdm(total=repeats, desc='Simulation Progress', unit=' repeat') if config['track']['bar'] else None

    for repeat in range(repeats):
        env.reset()

        while not game.stage_over():
            env.step()

            for player in players:
                player.update(type='current', data=env.state)

                player.select()

            intentions = utils.filterout(input=game.actions)
            actions, distinct, loser = env.coordinate(intentions)

            for action in actions:
                loser = env.apply(action, distinct=distinct, loser=loser)
            
            for player in players:
                env.reward(player=player, metrics=game.metrics)
                player.update(type='current', data=env.state)

                assert env.state[player.action.block.id].color.name == player.action.block.color.name, ("Color mismatch!")

            steps += 1

        for player in players:
            reward = player.R / steps
            rewards[player.type].append(reward)

        mistakes.append(env.num_conflicts)

        if pbar:
            metrics = {'Repeat': repeat + 1, 'Steps': steps, 'Mistakes': mistakes[repeat],
                       'CPU': f'{utils.get_cpu_usage():.2f}%', 'GPU': f'{utils.get_gpu_usage():.2f}%'}
            
            metrics.update({f'{type.capitalize()} Reward': f'{rewards[type][repeat]:.6f}' for type in types})

            pbar.set_postfix(metrics)
            pbar.update(1)

        logger.info(f"Repeat: {repeat + 1}, Steps: {steps}, " +
                    ", ".join([f"{type.capitalize()} Reward: {rewards[type][repeat]:.6f}" for type in types]) + ", " +
                    f"CPU: {utils.get_cpu_usage():.2f}%, GPU: {utils.get_gpu_usage():.2f}%")

        env.visualize(repeat=repeat, start=0, end=repeats, dir=('static', 'simulation', f'{game.title}', 'viz'))
    
    payoffs = {type: utils.aggregate(values=rewards[type]) for type in types}
    utils.save_json(data=payoffs, dir=('static', 'evaluation', 'payoffs'), name=game.title)

    if visualize:
        utils.plot(**__mistakes__(mistakes, repeats, game))
        utils.plot(**__rewards__(types, rewards, colors, repeats, game))
        utils.plot(**__stats__(players, top_k, steps, game))

def main():
    grid = Grid(**config['grid'])

    if 'human' in config:
        human = Player(**config['human'])
    else:
        human = None

    if 'robot' in config:
        robot = Player(**config['robot'])
    else:
        robot = None
    
    game = Game(env=grid, human=human, robot=robot, **config['game'])
    
    simulate(game=game, **config['simulate'])

if __name__ == '__main__':
    main()