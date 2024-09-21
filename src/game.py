from collections import namedtuple

class Game:
    def __init__(self, env, human, robot, gain, penalty, sanction, delay, title):
        self.env = env
        self.human = human
        self.robot = robot
        self.gain = gain
        self.penalty = penalty
        self.sanction = sanction
        self.delay = delay
        self.title = title

    @property
    def players(self):
        return [self.human, self.robot]

    @property
    def actions(self):
        Actions = namedtuple('Actions', ['human', 'robot'])

        human_action = self.human.action if self.human is not None else None
        robot_action = self.robot.action if self.robot is not None else None
        
        return Actions(human=human_action, robot=robot_action)
    
    @property
    def metrics(self):
        return (self.gain, self.penalty, self.sanction, self.delay)
    
    def load(self):
        self.env.load()
        for player in self.players:
            player.load(data=self.env.state) if player is not None else None
        
    def stage_over(self):
        for block in self.env.state:
            if block.is_hidden() or block.is_uncolored():
                return False
        return True