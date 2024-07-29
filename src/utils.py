import os
import yaml
import random
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
        raise ValueError("Unsupported structure type")
    
def encode(k, n):
    if k > n:
        raise ValueError("k should be less than or equal to n")
    
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
    if type == "robot":
        return "blue"
    elif type == "human":
        return "red"
    else:
        raise ValueError("Invalid type.")

def plot(values, labels, func, path, title, colors, width=0.2, ticks=(None,None), names=[None,None], marker=None, linestyle='solid'):
    x_label, y_label = labels
    x_ticks, y_ticks = ticks
    
    plt.figure(figsize=(12, 12))

    for i, pair in enumerate(values):
        x_values, y_values = pair

        if len(y_values) == 0:
            continue
        
        if func == plt.plot:
            func(x_values, y_values, color=colors[i], marker=marker, linestyle=linestyle, label=names[i])
        elif func == plt.bar:
            func(x_values, y_values, color=colors[i], width=width, label=names[i])
    
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if x_ticks is not None:
        plt.xticks(*x_ticks, rotation=45, ha="right")

    if y_ticks is not None:
        plt.yticks(*y_ticks, rotation=45, ha="right")

    if names != [None,None]:
        plt.legend()
        
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
    
def get_logger(level='DEBUG'):
    logger = logging.getLogger(__name__)

    level_name = logging.getLevelName(level)
    logger.setLevel(level_name)
    
    formatter = logging.Formatter('%(asctime)s:%(lineno)d:%(levelname)s:%(name)s:%(message)s')
    
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    return logger

def load_yaml(path):
    with open(path, 'r') as file:
        yaml_data = yaml.safe_load(file)
    
    return yaml_data

def get_adjacent_pos(row, col, direction):
    effect = {'left': (0, -1), 'right': (0, 1), 'up': (-1, 0), 'down': (1, 0)}

    row_effect, col_effect = effect[direction]
    
    return row + row_effect, col + col_effect

def aggregate(values, method='mean'):
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
        raise ValueError(f"Unsupported aggregation method: {method}")
    
def get_cpu_usage():
    return psutil.cpu_percent(interval=None)

def get_gpu_usage():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory * 100
    
    return 0