import os
import matplotlib.pyplot as plt

from . import utils

config = utils.load_yaml(path=utils.get_path(dir=(os.path.dirname(__file__), '..'), name='config.yaml'))

level = config['track']['logger']
path = utils.get_path(dir=('static', 'learning', config['game']['title']), name='loggings.pth')
logger = utils.get_logger(level=level, path=path)

def __stats__(base_dir, sub_dirs, top_k=20):
    labels = ('Action', 'Frequency')
    colors = ('blue', 'red')
    title = 'Statistics'
    func = plt.bar

    for dir in sub_dirs:
        search_dir = os.path.join(base_dir, dir)
        
        if not os.path.isdir(search_dir):
            continue
        
        for subfolder in os.listdir(search_dir):
            values, ticks, names = [], [], []

            subfolder_path = os.path.join(search_dir, subfolder)
            stats_data = utils.load_json(dir=subfolder_path, name='stats')

            for player_type, player_data in stats_data.items():
                actions = player_data[0]['actions'][:top_k]
                freqs = player_data[0]['freqs'][:top_k]

                x_values = list(range(top_k))
                x_ticks = (x_values, actions)

                values.append((x_values, freqs))
                ticks.append((x_ticks, None))
                names.append(player_type)

            game_title = os.path.basename(subfolder_path)
            output_path = utils.get_path(dir=('static', 'learning', game_title), name='statistics.png')

            utils.plot(values=values, labels=labels, func=func, path=output_path, title=title, colors=colors, ticks=ticks, names=names)

def main():
    __stats__(base_dir='static', sub_dirs=['learning', 'simulations'], top_k=20)

if __name__ == '__main__':
    main()
