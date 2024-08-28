import os
import yaml
import random
import pickle
import psutil
import torch
import logging
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

def get_size(struct):
    if isinstance(struct, (dict, list, tuple, set)):
        return len(struct)
    else:
        raise ValueError('Unsupported structure type')
    
def encode(k, n):
    if k > n:
        raise ValueError('k should be less than or equal to n')
    
    encoding = [0 for _ in range(n)]
    encoding[k - 1] = 1

    return encoding

def get_id(list, value):
    ids = [i for i, x in enumerate(list) if x == value]

    if ids:
        return random.choice(ids)
    else:
        return None

def get_path(dir, name):
    dir = dir if isinstance(dir, tuple) else (dir,)
    save_dir = os.path.join(*dir)

    os.makedirs(save_dir, exist_ok=True)
    
    return os.path.join(save_dir, name)

def get_color(type):
    if type == 'robot':
        return 'blue'
    elif type == 'human':
        return 'red'
    else:
        raise ValueError('Invalid type.')

def plot(values, labels, func, path, title, colors, width=0.2, ticks=[], names=[None, None], marker=None, linestyle='solid'):
    x_label, y_label = labels

    handles = []

    num_plots = len(values)
    total_width = num_plots * width
    offsets = [-total_width/2 + i * width + width/2 for i in range(num_plots)]

    plt.figure(figsize=(15, 15))
    ax = plt.gca()

    for i, pair in enumerate(values):
        x_values, y_values = pair

        if len(y_values) == 0:
            continue

        if ticks:
            if len(ticks) != len(values):
                raise ValueError("Ticks and values must have the same length")

            x_ticks, y_ticks = ticks[i]
        else:
            x_ticks, y_ticks = None, None

        if func == plt.plot:
            handle, = func(x_values, y_values, color=colors[i], marker=marker, linestyle=linestyle, label=names[i])
        elif func == plt.bar:
            x_values = [x + offsets[i] for x in x_values]
            handle = func(x_values, y_values, color=colors[i], width=width, label=names[i])

        handles.append(handle)

        if x_ticks:
            if i == 0:
                ax.set_xticks(x_ticks[0])
                ax.set_xticklabels(x_ticks[1], rotation=45, ha='right')
            else:
                ax2 = ax.twiny()

                ax2.set_xticks(x_ticks[0])
                ax2.set_xticklabels(x_ticks[1])

                ax2.spines['top'].set_position(('outward', 50 * i))
                ax2.set_xlim(ax.get_xlim())

                plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

        if y_ticks:
            if i == 0:
                ax.set_yticks(y_ticks[0])
                ax.set_yticklabels(y_ticks[1])
            else:
                ax2 = ax.twinx()
                
                ax2.set_yticks(y_ticks[0])
                ax2.set_yticklabels(y_ticks[1])
                
                ax2.spines['right'].set_position(('outward', 50 * i))
                ax2.set_ylim(ax.get_ylim())

    plt.title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if names:
        plt.legend(handles=handles, labels=names)

    plt.grid(True)
    plt.savefig(path)
    plt.close()

def filterout(input, target=None):
    if isinstance(input, list):
        target_func = lambda x: x is not target

        return list(filter(target_func, input))
    
    elif isinstance(input, tuple) and hasattr(input, '_fields'):
        class_name = input.__class__.__name__
        new_items = [(key, value) for key, value in input._asdict().items() if value is not target]

        fields = [item[0] for item in new_items]
        values = [item[1] for item in new_items]

        output = namedtuple(class_name, fields)

        return output(*values)
    
def get_logger(level='DEBUG', path=''):
    logger = logging.getLogger(__name__)

    level_name = logging.getLevelName(level)
    logger.setLevel(level_name)
    
    formatter = logging.Formatter('%(asctime)s:%(lineno)d:%(levelname)s:%(name)s:%(message)s')
    
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

def load_yaml(path):
    with open(path, 'r') as file:
        yaml_data = yaml.safe_load(file)
    
    return yaml_data

def get_adjacent_pos(row, col, direction):
    effect = {'left': (0, -1), 'right': (0, 1), 'up': (-1, 0), 'down': (1, 0)}

    row_effect, col_effect = effect[direction]
    
    return row + row_effect, col + col_effect

def aggregate(values, weights=None, method='mean', remove_zeros=True):
    values = np.array(values)
    
    if weights is not None:
        weights = np.array(weights)

        if remove_zeros:
            mask = weights != 0
            values = values[mask]
            weights = weights[mask]

        values = values * weights

    if method == 'mean':
        return np.mean(values)
    
    elif method == 'sum':
        return np.sum(values)
    
    elif method == 'median':
        return np.median(values)
    
    elif method == 'min':
        return np.min(values)
    
    elif method == 'max':
        return np.max(values)
    
    else:
        raise ValueError(f'Unsupported aggregation method: {method}')
    
def get_cpu_usage():
    return psutil.cpu_percent(interval=None)

def get_gpu_usage():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory * 100
    
    return 0

def is_last(current, final, k=-1):
    # current will never exceed final during normal execution, so any value of current that is greater than or equal to final i.e., final + 1, is effectively out of bounds.
    return current >= (final - k)

def load_model(path):
    model = torch.load(path)
    model.eval()

    return model

def save_pickle(data, name):
    path = get_path(dir='static', name=f'{name}.pkl')

    with open(path, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(name):
    path = get_path(dir='static', name=f'{name}.pkl')

    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data

    except FileNotFoundError:
        return None