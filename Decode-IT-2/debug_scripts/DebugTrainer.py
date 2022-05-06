import gym

from Trainer import RfTrainer
from rf_model import RfModel

env = gym.make("FrozenLake-v0")
agent =  RfModel()
trainer = RfTrainer(env=env,agent=None)