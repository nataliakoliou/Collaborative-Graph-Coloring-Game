import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from . import utils
from .game import Game
from .grid import Grid
from .player import Player

logger = utils.get_logger(level='DEBUG')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger.info(f'Device is {device}')

def qlearn(game, repeats, epsilon, cutoff, visualize, phase):
    env = game.env
    players = utils.filterout(input=game.players)
    types = [player.type for player in players]
    colors = [player.color for player in players]

    max_exploration_steps = cutoff * repeats
    decay = round(1/max_exploration_steps, 10)

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

        env.visualize(repeat=repeat, start=0, end=repeats, title=game.title, phase=phase)

    for player in players:
        action_freqs[player.type] = [action.times['Exploitation']/steps for action in player.space]

        path = utils.get_path(dir=("models", f"{player.type}"), name=f"{game.title}.pth")
        model = player.policy_net.state_dict()
        torch.save(model, path)

    if visualize:

        repeats_lst = list(range(repeats))

        utils.plot(values=[(repeats_lst, mistakes)], 
                   labels=('Repeat', 'Mistakes'), 
                   func=plt.bar,
                   path=utils.get_path(dir=("static", f"{game.title}", phase), name="learn_mistakes.png"), 
                   title='Mistakes Plot', 
                   colors=['green']
                   )
        
        utils.plot(values=[(repeats_lst, losses[type]) for type in types],
                   labels=('Step', 'Loss'),
                   func=plt.plot,
                   path=utils.get_path(dir=("static", f"{game.title}", phase), name="losses.png"),  
                   title='Loss Curve',
                   colors=colors,
                   names=types
                   )

        utils.plot(values=[(list(range(len(action_freqs[type]))), action_freqs[type]) for type in types], 
                   labels=('Action', 'Frequency'),
                   func=plt.bar,
                   path=utils.get_path(dir=("static", f"{game.title}", phase), name=f"learn_exploit_stats.png"), 
                   title='Exploitation Statistics', 
                   colors=colors
                   )

def main():
    config = utils.load_yaml(path=utils.get_path(dir=(os.path.dirname(__file__), '..'), name='config.yaml'))
    
    grid = Grid(**config['grid'])
    human = Player(**config['human'])
    
    game = Game(env=grid, human=human, robot=None)
    
    qlearn(game=game, **config['qlearn'])

if __name__ == "__main__":
    main()