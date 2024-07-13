import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

from . import utils
from .game import Game
from .grid import Grid
from .player import Player

logger = utils.get_logger(level='DEBUG')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger.info(f'Device is {device}')

def qlearn(game, repeats, epsilon, cutoff, visualize=True):
    env = game.env
    players = utils.filterout(input=game.players)
    types = [player.type for player in players]
    colors = [player.color for player in players]

    explore = int(cutoff * repeats)
    decay = round(1/explore, 10)

    losses = {type: [] for type in types}
    action_freqs = {type: [] for type in types}
    mistakes = []
    steps = 0

    game.load()

    for repeat in range(repeats):
        env.reset()

        while not game.stage_over():
            env.step()

            for player in players:
                player.update(type="current", data=env.state)

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

                player.update(type="next", data=env.state)
                player.expand_memory()

                player.update(type="current", data=env.state)
                player.optimize()

                player.update(type="net")

            steps += 1

        for player in players:
            losses[player.type].append(player.L / steps)

        mistakes.append(env.num_constraints)
        epsilon = max(epsilon - decay, 0)

        logger.info(f"Repeat: {repeat + 1} ~ Steps: {steps} ~ " + " ~ ".join([f"Losses ({type})={losses[type][repeat]:.6f}" for type in types]) + 
                    f" ~ Mistakes: {mistakes[repeat]} ~ Epsilon: {epsilon:.6f}")

    for player in players:
        action_freqs[player.type] = [action.times['Exploitation']/steps for action in player.space]

        path = utils.get_path(folder=("models", f"{player.type}"), name=f"{game.title}.pth")
        model = player.policy_net.state_dict()
        torch.save(model, path)

    if visualize:

        repeats_lst = list(range(repeats))

        utils.plot(values=[(repeats_lst, mistakes)], 
                   labels=('Repeat', 'Mistakes'), 
                   func=plt.bar,
                   path=utils.get_path(folder=("static", f"{game.title}"), name="learn_mistakes.png"), 
                   title='Mistakes Plot', 
                   colors=['green']
                   )
        
        utils.plot(values=[(repeats_lst, losses[type]) for type in types],
                   labels=('Step', 'Loss'),
                   func=plt.plot,
                   path=utils.get_path(folder=("static", f"{game.title}"), name="losses.png"),  
                   title='Loss Curve',
                   colors=colors,
                   names=types
                   )

        utils.plot(values=[(list(range(len(action_freqs[type]))), action_freqs[type]) for type in types], 
                   labels=('Action', 'Frequency'),
                   func=plt.bar,
                   path=utils.get_path(folder=("static", f"{game.title}"), name=f"learn_exploit_stats.png"), 
                   title='Exploitation Statistics', 
                   colors=colors
                   )

def main():
    config = utils.load_yaml(path=os.path.join(os.path.dirname(__file__), '..', 'config.yaml'))
    
    grid = Grid(rows=config['grid']['rows'], cols=config['grid']['cols'], merge=config['grid']['merge'], 
                minR=config['grid']['minR'], wR=config['grid']['wR'])
    
    player = Player(type=config['player']['type'], style=config['player']['style'], 
                    model=config['player']['model'], criterion=config['player']['criterion'], 
                    optimizer=config['player']['optimizer'], tau=config['player']['tau'], 
                    batch_size=config['player']['batch_size'], gamma=config['player']['gamma'])
    
    game = Game(env=grid, human=player, robot=None)
    
    qlearn(game=game, repeats=config['qlearn']['repeats'], epsilon=config['qlearn']['epsilon'], 
           cutoff=config['qlearn']['cutoff'], visualize=config['qlearn']['visualize'])

if __name__ == "__main__":
    main()