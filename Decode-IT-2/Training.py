import time

import gym
import torch
import torch.optim as optim

from DeepAgent import DeepAgent
from EnvManager import EnvManager
from Enviornment import Enviornment
from Qvalues import Qvalues
from Trainer import RfTrainer
#from models.D import DCNN
from models.DCNN import DCNN
from models.DCNN_deep import DCNN_DEEP
from models.DQN import DQN

# params


learning_rate = 0.001
exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0
exploration_decay_rate = 0.005
num_episodes = 100000
target_update = 10
max_steps_per_episode = 10000
gamma = 0.999
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "gpu"
#Qvalues.device=device
# objects
#env = gym.make('CartPole-v0')
env = Enviornment()
env.reset()
em = EnvManager(env=env, device=device)
num_actions = em.num_actions_avaiable()
width = em.get_screen_width()
height = em.get_screen_height()
name = "High_rewards_high_punisment"
model = DQN(img_width=width, img_height=height, channels=4, actions_space_size=num_actions)
model = DCNN(img_width=width, img_height=height, channels=1, actions_space_size=num_actions)
#model = DCNN_DEEP(img_width=width, img_height=height, channels=1, actions_space_size=num_actions)
model = model.to(device)
zero_state = em.get_state()
with torch.no_grad():
    test = model(zero_state)
optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)
agent = DeepAgent(deep_model=model, optimizer=optimizer, replay_memory_size=15000, device=device, name=name)

agent.load_model()
batch_size= 1024
env.ready=True
Trainer = RfTrainer(em=em,
                    device=device,
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
while env.pause or not env.BotController.activate:
    time.sleep(0.25)
    #print("Waiting for env")
    print(env.BotController.human_pressed_key)
Trainer.train()
