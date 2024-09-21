import os
from ruamel.yaml import YAML

yaml = YAML()

BASE_DIR = os.path.join(os.path.dirname(__file__), '../settings')
LEARNING_DIR = os.path.join(BASE_DIR, 'learning')
SIMULATION_DIR = os.path.join(BASE_DIR, 'simulation')

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.load(file)

def write_yaml(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as file:
        yaml.dump(data, file)

def load_common_track_game_grid(sample_file):
    sample_yaml = load_yaml(sample_file)
    return {
        'track': sample_yaml.get('track', {}),
        'game': sample_yaml.get('game', {}),
        'grid': sample_yaml.get('grid', {}),
    }

def generate_simulation_files():
    human_dir = os.path.join(LEARNING_DIR, 'human')
    robot_dir = os.path.join(LEARNING_DIR, 'robot')
    
    human_files = [f for f in os.listdir(human_dir) if f.endswith('.yaml')]
    robot_files = [f for f in os.listdir(robot_dir) if f.endswith('.yaml')]

    sample_learning_file = os.path.join(human_dir, 'I.yaml')
    common_track_game_grid = load_common_track_game_grid(sample_learning_file)

    for human_file in human_files:
        for robot_file in robot_files:
            human_style = os.path.splitext(human_file)[0]
            robot_style = os.path.splitext(robot_file)[0]

            human_yaml = load_yaml(os.path.join(human_dir, human_file))
            robot_yaml = load_yaml(os.path.join(robot_dir, robot_file))

            sim_title = f"{human_style}_x_{robot_style}"

            simulation_yaml = {
                'track': common_track_game_grid['track'],
                'game': {
                    **common_track_game_grid['game'],
                    'title': sim_title,
                },
                'grid': common_track_game_grid['grid'],
                'human': human_yaml['human'],
                'robot': robot_yaml['robot'],
                'simulate': {
                    'repeats': 5000,
                    'visualize': True,
                    'top_k': 20
                }
            }

            output_file = os.path.join(SIMULATION_DIR, f"{sim_title}.yaml")
            write_yaml(simulation_yaml, output_file)

            print(f"Generated simulation file: {sim_title}.yaml")

def main():
    print("Generating simulation YAMLs...")
    generate_simulation_files()
    print("Simulation YAML generation completed!")

if __name__ == "__main__":
    main()