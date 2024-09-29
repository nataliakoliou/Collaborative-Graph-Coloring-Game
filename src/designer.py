import os
import matplotlib.pyplot as plt

from . import utils

config = utils.load_yaml(path=utils.get_path(dir=(os.path.dirname(__file__), '..'), name='config.yaml'))

level = config['track']['logger']
path = utils.get_path(dir=('static', 'learning', config['game']['title']), name='loggings.pth')
logger = utils.get_logger(level=level, path=path)

def get_learning_logs(base_dir, include):
    logs = {"repeat": {}, "loss": {}, "reward": {}, "mistakes": {}, "epsilon": {}, "cpu": {}, "gpu": {}}

    for game_title in os.listdir(base_dir):
        if game_title not in include:
            continue

        game_path = os.path.join(base_dir, game_title)
        if not os.path.isdir(game_path):
            continue

        log_path = os.path.join(game_path, 'loggings.pth')

        with open(log_path, 'r') as log_file:
            log_lines = log_file.readlines()

        filtered_lines = []
        for log_entry in log_lines:
            if 'Repeat' in log_entry:
                filtered_lines.append(log_entry)

                if 'Repeat: 10000' in log_entry:
                    break

        with open(log_path, 'w') as log_file:
            log_file.writelines(filtered_lines)

        repeats, losses, rewards, mistakes, epsilon, cpu, gpu = [], [], [], [], [], [], []

        for log_entry in filtered_lines:
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

    utils.save_json(data=logs, dir=base_dir, name='full_loggings')

    return logs

def get_simulation_logs(base_dir, include):
    logs = {"repeat": {}, "human_reward": {}, "robot_reward": {}, "mistakes": {}, "cpu": {}, "gpu": {}}

    for game_title in os.listdir(base_dir):
        if game_title not in include:
            continue

        game_path = os.path.join(base_dir, game_title)
        if not os.path.isdir(game_path):
            continue

        log_path = os.path.join(game_path, 'loggings.pth')

        with open(log_path, 'r') as log_file:
            log_lines = log_file.readlines()

        filtered_lines = []
        for log_entry in log_lines:
            if 'Repeat' in log_entry:
                filtered_lines.append(log_entry)

                if 'Repeat: 2000' in log_entry:
                    break

        with open(log_path, 'w') as log_file:
            log_file.writelines(filtered_lines)

        repeats, human_rewards, robot_rewards, mistakes, cpu, gpu = [], [], [], [], [], []

        for log_entry in filtered_lines:
            log_data = log_entry.split('Repeat:')[1]
            parts = log_data.split(',')

            repeats.append(int(parts[0].strip())) 
            mistakes.append(int(parts[2].split(':')[1].strip()))
            human_rewards.append(float(parts[3].split(':')[1].strip()))
            robot_rewards.append(float(parts[4].split(':')[1].strip()))
            cpu.append(float(parts[5].split(':')[1].strip().replace('%', '')))
            gpu.append(float(parts[6].split(':')[1].strip().replace('%', '')))

        logs["repeat"][game_title] = repeats
        logs["human_reward"][game_title] = human_rewards
        logs["robot_reward"][game_title] = robot_rewards
        logs["mistakes"][game_title] = mistakes
        logs["cpu"][game_title] = cpu
        logs["gpu"][game_title] = gpu

    utils.save_json(data=logs, dir=base_dir, name='full_loggings')

    return logs

def __learning_logs__(include, title, labels, colors, func, shift):
    base_dir = 'static/learning'
    logs = get_learning_logs(base_dir, include)
    
    x_key = labels[0].lower()
    y_key = labels[1].lower()
    
    values = []
    for key in include:
        x_values = logs[x_key][key]
        y_values = logs[y_key][key]

        values.append((x_values, y_values))

    include_str = ''.join(include)
    output_path = utils.get_path(dir=base_dir, name=f'{y_key}_{include_str}.png')
    
    utils.plot(values=values, labels=labels, func=func, path=output_path, title=title, colors=colors, names=include, shift=shift)

def __simulation_logs__(include, title, labels, colors, func, shift):
    base_dir = 'static/simulation'
    logs = get_simulation_logs(base_dir, include)
    
    x_key = labels[0].lower()
    y_key = labels[1].lower()

    values = []
    for key in include:
        x_values = logs[x_key][key]

        if 'reward' in y_key:
            y_values_human = logs['human_reward'][key]
            y_values_robot = logs['robot_reward'][key]
            
            values.append((x_values, y_values_human))
            values.append((x_values, y_values_robot))
        else:
            y_values = logs[y_key][key]
            values.append((x_values, y_values))

    include_str = '_'.join(include)
    output_path = utils.get_path(dir=base_dir, name=f'{y_key}_{include_str}.png')
    names = [item for item in include for _ in range(2)]
    
    utils.plot(values=values, labels=labels, func=func, path=output_path, title=title, colors=colors, names=names, shift=shift) 

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
            output_path = utils.get_path(dir=(search_dir, game_title), name='statistics.png')

            utils.plot(values=values, labels=labels, func=func, path=output_path, title=title, colors=colors, ticks=ticks, names=names)
    
def main():
    # 1 statistic plot per game either, both in learning and simulation
    """__stats__(base_dir='static', 
              sub_dirs=['learning', 'simulation'], 
              top_k=20,
              title='Statistics',
              labels=('Action', 'Frequency'),
              colors=['blue', 'red'],
              func=plt.bar)"""

    # 1 loss plot for all defined game titles during learning
    """__learning_logs__(include=['CA', 'A', 'CAM'],
                      title='Loss Curves', 
                      labels=('Repeat', 'Loss'), 
                      colors=['blue', 'red', 'green'],
                      func=plt.plot,
                      shift=0.01)"""
    
    # 1 rewards plot for all defined game titles during simulation
    __simulation_logs__(include=['I_x_C', 'A_x_CA'], 
                        title='Reward Curve', 
                        labels=('Repeat', 'Reward'), 
                        colors=['blue', 'blue', 'red', 'red'],
                        func=plt.plot,
                        shift=0.01)

if __name__ == '__main__':
    main()