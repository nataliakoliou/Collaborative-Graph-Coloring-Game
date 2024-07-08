import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import logging

import utils
from game import Game
from grid import Grid
from player import Player

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(lineno)d:%(levelname)s:%(name)s:%(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f'Device is {device}')

dir = '/content/drive/MyDrive/Repositories/custom_spiel'

def qlearn(game, repeats, epsilon, cutoff, visualize=True):
    env = game.env
    players = utils.filterout(input=game.players)
    types = [player.type for player in players]
    colors = [player.color for player in players]

    explore = int(cutoff * repeats)
    decay = round(1/explore)

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

        mistakes.append(env.constraints)
        epsilon = max(epsilon - decay, 0)

        logger.info(f"Repeat: {repeat + 1} ~ Steps: {steps} ~ " + " ~ ".join([f"Losses ({type})={losses[type][repeat]:.6f}" for type in types]) + f" ~ Mistakes: {mistakes[repeat]}")

    for player in players:
        action_freqs[player.type] = [action.times['Exploitation']/steps for action in player.space]

        path = utils.get_path(dir=dir, folder=("models", f"{player.type}"), name=f"{game.title}.pth")
        model = player.policy_net.state_dict()
        torch.save(model, path)

    if visualize:

        utils.plot(values=[(range(repeats), mistakes)], 
                   labels=('Repeat', 'Mistakes'), 
                   func=plt.bar,
                   path=utils.get_path(dir=dir, folder=("static", f"{game.title}"), name="learn_mistakes.png"), 
                   title='Mistakes Plot', 
                   colors=['green'])
        
        utils.plot(values=[(range(repeats), losses[type]) for type in types],
                   labels=('Step', 'Loss'),
                   func=plt.plot,
                   path=utils.get_path(dir=dir, folder=("static", f"{game.title}"), name="losses.png"),  
                   title='Loss Curve',
                   colors=colors,
                   names=types)

        utils.plot(values=[(list(range(len(action_freqs[type]))), action_freqs[type]) for type in types], 
                   labels=('Action', 'Frequency'),
                   func=plt.bar,
                   path=utils.get_path(dir=dir, folder=("static", f"{game.title}"), name=f"learn_exploit_stats.png"), 
                   title='Exploitation Statistics', 
                   colors=colors)

def main():
    grid = Grid(rows=2, cols=2, merge=0.2, minR=2, wR=0.2)
    # grid = Grid(rows=4, cols=4, merge=0.3, minR=2, wR=0.2)

    human = Player(type='human', 
                   style='random',
                   model='HDQN', 
                   criterion=nn.SmoothL1Loss(beta=1.0), 
                   optimizer='AdamW', 
                   lr=1e-3, 
                   tau=5e-3, 
                   batch_size=8, 
                   gamma=0, 
                   weight_decay=1e-5)
    
    game = Game(env=grid, human=human, robot=None)
    
    qlearn(game=game, repeats=100, epsilon=1, cutoff=0.9, visualize=True)

if __name__ == "__main__":
    main()