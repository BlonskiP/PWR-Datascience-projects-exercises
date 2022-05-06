import time

import gym
import torch
import torch.optim as optim

from DeepAgent import DeepAgent
from EnvManager import EnvManager
from Enviornment import Enviornment
from Qvalues import Qvalues
from Trainer import RfTrainer
from models.DCNN import DCNN
from models.DQN import DQN




learning_rate = 0.001

max_exploration_rate = 1
min_exploration_rate = 0.1
exploration_decay_rate = 0.0001
num_episodes = 1000
target_update = 10
max_steps_per_episode = 10000
gamma = 0.9999
replay_memory_size = 100000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"
Qvalues.device=device
# objects
env = gym.make("Acrobot-v1")
env.reset()
em = EnvManager(env=env, device=device,gym=True)
num_actions = em.num_actions_avaiable()
width = em.get_screen_width()
height = em.get_screen_height()
name = "Acrobot-v1-4"
#model = DQN(img_width=width, img_height=height, channels=3, actions_space_size=num_actions)
model = DCNN(img_width=width, img_height=height, channels=1, actions_space_size=num_actions,name=name)
model=model.to(device)
zero_state = em.get_state()
test = model(zero_state)
optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)
agent = DeepAgent(deep_model=model, optimizer=optimizer, replay_memory_size=replay_memory_size, device=device, name=name)

#agent.load_model()
batch_size= 600
Trainer = RfTrainer(em=em,
                    agent=agent,
                    learning_rate=learning_rate,
                    min_exploration_rate=min_exploration_rate,
                    max_exploration_rate=max_exploration_rate,
                    exploration_decay=exploration_decay_rate,
                    gamma=gamma,
                    num_episodes=num_episodes,
                    max_steps_per_episode=max_steps_per_episode,
                    target_update=target_update,
                    batch_size=batch_size,
                    openAI_gym=True)
Trainer.train()
