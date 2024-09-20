import os
import matplotlib.pyplot as plt

from . import utils

config = utils.load_yaml(path=utils.get_path(dir=(os.path.dirname(__file__), '..'), name='config.yaml'))

level = config['track']['logger']
path = utils.get_path(dir=('static', 'learning', config['game']['title']), name='loggings.pth')
logger = utils.get_logger(level=level, path=path)

def retrieve_logs(base_dir='static/learning'):
    logs = {"repeat": {},"loss": {}, "reward": {}, "mistakes": {},"epsilon": {},"cpu": {},"gpu": {}}

    for game_title in os.listdir(base_dir):
        game_path = os.path.join(base_dir, game_title)

        if not os.path.isdir(game_path) or len(game_title) != 1:
            continue

        log_path = os.path.join(game_path, 'loggings.pth')

        with open(log_path, 'r') as log_file:
            log_lines = log_file.readlines()

        repeats, losses, rewards, mistakes, epsilon, cpu, gpu = [], [], [], [], [], [], []

        for log_entry in log_lines:
            if 'Repeat' in log_entry:
                log_data = log_entry.split('Repeat:')[1]

                parts = log_data.split(',')

                repeats.append(int(parts[0].strip())) 
                losses.append(float(parts[2].split(':')[1].strip()))
                rewards.append(float(parts[3].split(':')[1].strip()))
                mistakes.append(int(parts[4].split(':')[1].strip()))
                epsilon.append(float(parts[5].split(':')[1].strip()))
                cpu.append(float(parts[6].split(':')[1].strip().replace('%', '')))
                gpu.append(float(parts[7].split(':')[1].strip().replace('%', '')))

        logs["repeat"][game_title] = repeats
        logs["loss"][game_title] = losses
        logs["reward"][game_title] = rewards
        logs["mistakes"][game_title] = mistakes
        logs["epsilon"][game_title] = epsilon
        logs["cpu"][game_title] = cpu
        logs["gpu"][game_title] = gpu

    utils.save_json(data=logs, dir=('static', 'learning'), name='full_loggings')

def __logs__(include, title, labels, colors, func):
    logs = utils.load_json(dir=('static', 'learning'), name='full_loggings')
    
    x_key = labels[0].lower()
    y_key = labels[1].lower()

    max_length = max(len(logs[x_key][game]) for game in include)
    
    values = []
    for key in include:
        x_values = logs[x_key][key]
        y_values = logs[y_key][key]

        padded_x_values = x_values + [None] * (max_length - len(x_values))
        padded_y_values = y_values + [None] * (max_length - len(y_values))

        values.append((padded_x_values, padded_y_values))

    include_str = ''.join(include)
    output_path = utils.get_path(dir=('static', 'learning'), name=f'{y_key}_{include_str}.png')
    
    utils.plot(values=values, labels=labels, func=func, path=output_path, title=title, colors=colors, names=include)

def __stats__(base_dir, sub_dirs, top_k, title, labels, colors, func):
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
    retrieve_logs(base_dir='static/learning')

    __stats__(base_dir='static', 
              sub_dirs=['learning', 'simulations'], 
              top_k=20,
              title='Statistics',
              labels=('Action', 'Frequency'),
              colors=['blue', 'red'],
              func=plt.bar)

    __logs__(include=['I', 'C'],
             title='Loss Curves', 
             labels=('Repeat', 'Loss'), 
             colors=['blue', 'red'],
             func=plt.plot)

if __name__ == '__main__':
    main()