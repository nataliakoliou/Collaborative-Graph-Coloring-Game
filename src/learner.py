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

def __stats__(players, top_k, steps, game):
    values, ticks, colors, names = [], [], [], []
    labels = ('Action', 'Frequency')
    func = plt.bar
    path = utils.get_path(dir=('static', 'learning', f'{game.title}'), name='statistics.png')
    title = 'Statistics'

    stats = {player.type: [] for player in players}

    for player in players:
        actions_and_freqs = [
            (f'(B{action.block.id}, {action.color.name})', action.times['Exploitation'] / steps)
            for action in player.space
        ]
        
        actions_and_freqs.sort(key=lambda x: x[1], reverse=True)

        all_actions, all_freqs = zip(*actions_and_freqs)
        stats[player.type].append({'actions': all_actions, 'freqs': list(all_freqs)})

        top_actions_and_freqs = actions_and_freqs[:top_k]
        actions, freqs = zip(*top_actions_and_freqs)
        
        x_values = list(range(top_k))
        x_ticks = (x_values, list(actions))
        
        values.append((x_values, list(freqs)))
        ticks.append((x_ticks, None))
        colors.append(player.color)
        names.append(player.type)
    
    utils.save_json(data=stats, dir=('static', 'learning', f'{game.title}'), name='stats')
    
    return {'values': values, 'labels': labels, 'func': func, 'path': path, 'title': title, 'colors': colors, 'ticks': ticks, 'names': names}

def __rewards__(types, rewards, colors, repeats, game):
    values = [(list(range(repeats)), rewards[type]) for type in types]
    labels = ('Repeat', 'Reward')
    func = plt.plot
    path = utils.get_path(dir=('static', 'learning', f'{game.title}'), name='rewards.png')
    title = 'Reward Curve'

    return {'values': values, 'labels': labels, 'func': func, 'path': path, 'title': title, 'colors': colors, 'names': types}

def __losses__(types, losses, colors, repeats, game):
    values = [(list(range(repeats)), losses[type]) for type in types]
    labels = ('Repeat', 'Loss')
    func = plt.plot
    path = utils.get_path(dir=('static', 'learning', f'{game.title}'), name='losses.png')
    title = 'Loss Curve'

    return {'values': values, 'labels': labels, 'func': func, 'path': path, 'title': title, 'colors': colors, 'names': types}

def __mistakes__(mistakes, repeats, game):
    values = [(list(range(repeats)), mistakes)]
    labels = ('Repeat', 'Mistakes')
    func = plt.plot
    path = utils.get_path(dir=('static', 'learning', f'{game.title}'), name='mistakes.png')
    title = 'Mistakes'
    colors = ['green']

    return {'values': values, 'labels': labels, 'func': func, 'path': path, 'title': title, 'colors': colors}

def qlearn(game, repeats, epsilon, cutoff, patience, visualize, top_k, config, logger):
    env = game.env
    players = utils.filterout(input=game.players)
    types = [player.type for player in players]
    colors = [player.color for player in players]

    max_explore = math.ceil(cutoff * repeats)
    decay = round(1/max_explore, 10)

    losses = {type: [] for type in types}
    rewards = {type: [] for type in types}
    best_losses = {type: float('inf') for type in types}
    no_improvement = {type: 0 for type in types}
    mistakes = []
    steps = 0

    game.load()

    pbar = tqdm(total=repeats, desc='Learning Progress', unit=' repeat') if config['track']['bar'] else None

    for repeat in range(repeats):
        env.reset()

        while not game.stage_over():
            env.step()

            for player in players:
                player.update(type='current', data=env.state)

                if np.random.rand() < epsilon:
                    player.explore()
                else:
                    player.exploit()

            intentions = utils.filterout(input=game.actions)
            actions, distinct, loser = env.coordinate(intentions)

            for action in actions:
                loser = env.apply(action, distinct=distinct, loser=loser)
            
            for player in players:
                env.reward(player=player, metrics=game.metrics)

                player.update(type='next', data=env.state)
                player.expand_memory()

                player.update(type='current', data=env.state)
                player.optimize()

                player.update(type='net')

                assert env.state[player.action.block.id].color.name == player.action.block.color.name, ("Color mismatch!")

            steps += 1

        for player in players:
            loss = player.L / steps
            losses[player.type].append(loss)

            reward = player.R / steps
            rewards[player.type].append(reward)

            if repeat >= max_explore:
                if loss < best_losses[player.type]:
                    best_losses[player.type] = loss
                    no_improvement[player.type] = 0
                else:
                    no_improvement[player.type] += 1

        mistakes.append(env.num_conflicts)
        epsilon = max(epsilon - decay, 0)

        if pbar:
            metrics = {'Repeat': repeat + 1, 'Steps': steps, 'Mistakes': mistakes[repeat], 'Epsilon': f'{epsilon:.6f}',
                       'CPU': f'{utils.get_cpu_usage():.2f}%', 'GPU': f'{utils.get_gpu_usage():.2f}%'}
            
            metrics.update({f'{type.capitalize()} Loss': f'{losses[type][repeat]:.6f}' for type in types})
            metrics.update({f'{type.capitalize()} Reward': f'{rewards[type][repeat]:.6f}' for type in types})

            pbar.set_postfix(metrics)
            pbar.update(1)

        logger.info(f"Repeat: {repeat + 1}, Steps: {steps}, " +
                    ", ".join([f"{type.capitalize()} Loss: {losses[type][repeat]:.6f}" for type in types]) + ", " +
                    ", ".join([f"{type.capitalize()} Reward: {rewards[type][repeat]:.6f}" for type in types]) + ", " +
                    f"Mistakes: {mistakes[repeat]}, Epsilon: {epsilon:.6f}, " +
                    f"CPU: {utils.get_cpu_usage():.2f}%, GPU: {utils.get_gpu_usage():.2f}%")
        
        env.visualize(repeat=repeat, start=0, end=repeats, dir=('static', 'learning', f'{game.title}', 'viz'))
        
        if repeat >= max_explore:
            if any(no_improvement[type] >= patience for type in types):
                logger.info(f'Early stopping triggered after {repeat + 1} repeats with no improvement.')
                repeats = repeat + 1
                break

    for player in players:
        path = utils.get_path(dir=('models', f'{player.type}'), name=f'{player.style.name}.pth')
        model = player.policy_net.state_dict()
        torch.save(model, path)

    if visualize:
        utils.plot(**__mistakes__(mistakes, repeats, game))
        utils.plot(**__losses__(types, losses, colors, repeats, game))
        utils.plot(**__rewards__(types, rewards, colors, repeats, game))
        utils.plot(**__stats__(players, top_k, steps, game))

def main():
    dir = os.path.join(os.path.dirname(__file__), '..', 'settings', 'learning', 'human')
    files = [f for f in os.listdir(dir) if f.endswith('.yaml')]

    for file in files:
        path = utils.get_path(dir=dir, name=file)
        config = utils.load_yaml(path=path)

        level = config['track']['logger']
        path = utils.get_path(dir=('static', 'learning', config['game']['title']), name='loggings.pth')
        logger = utils.get_logger(level=level, path=path)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'Device is {device}')

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
        
        qlearn(game=game, **config['qlearn'], config=config, logger=logger)

if __name__ == '__main__':
    main()