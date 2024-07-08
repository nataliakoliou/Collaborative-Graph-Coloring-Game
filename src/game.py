import random
import utils
from torch.nn import SmoothL1Loss, MSELoss, L1Loss
from collections import namedtuple

class Game:
    def __init__(self, env, human, robot, gain=1, penalty=-2, sanction=-10):
        self.env = env
        self.human = human
        self.robot = robot
        self.gain = gain
        self.penalty = penalty
        self.sanction = sanction

    @property
    def players(self):
        return [self.human, self.robot]
    
    @property
    def title(self):
        human_desc = f"{self.human.type}_{self.human.style}" if self.human else ""
        robot_desc = f"{self.robot.type}_{self.robot.style}" if self.robot else ""

        return f"{human_desc}_{robot_desc}".strip('_')

    @property
    def actions(self):
        Actions = namedtuple('Actions', ['human', 'robot'])

        human_action = self.human.action if self.human is not None else None
        robot_action = self.robot.action if self.robot is not None else None
        
        return Actions(human=human_action, robot=robot_action)
    
    @property
    def metrics(self):
        return (self.gain, self.penalty, self.sanction)
    
    def load(self):
        self.env.load()
        for player in self.players:
            player.load(data=self.env.state) if player is not None else None
        
    def stage_over(self):
        for block in self.env.state:
            if block.is_hidden() or block.is_uncolored():
                return False
        return True