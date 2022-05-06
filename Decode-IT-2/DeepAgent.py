import copy
import random
import numpy as np
import torch
import pickle
from pathlib import Path

class DeepAgent:
    def __init__(self, deep_model,optimizer, replay_memory_size=100, device="cpu",name="model"):
        self.replay_memory = []
        self.human_guidelines = []
        self.name =name
        self.replay_memory_size = replay_memory_size
        self.device = device
        self.model = deep_model.to(device)
        self.target_network = copy.copy(self.model)
        self.set_target_network()
        self.optimizer=optimizer
        self.memory_path = Path(f"{self.name}+_mem.pkl")
        self.memory_human_path = Path(f"human_mem.pkl")
        self.load_memory()
        self.memory_upload_counter =0
        self.memory_human_upload_counter = 0

    def save_memory(self,type="bot"):
        if type=="bot":
            with open(self.memory_path, 'wb') as handle:
                pickle.dump(self.replay_memory, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(self.memory_human_path, 'wb') as handle:
                pickle.dump(self.human_guidelines, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def upload_memory(self):
        if self.memory_upload_counter % 1001 == 0:
            self.save_memory(type="bot")
            self.memory_upload_counter = 0
        if self.memory_human_upload_counter % 200 == 0:
            self.save_memory(type="human")
            self.memory_human_upload_counter = 0

    def load_memory(self):
        if self.memory_path.exists():
            with open(self.memory_path, 'rb') as handle:
               self.replay_memory = pickle.load(handle)
               if len(self.replay_memory) > self.replay_memory_size:
                   self.replay_memory=self.replay_memory[:self.replay_memory_size]

        if self.memory_human_path.exists():
            with open(self.memory_human_path, 'rb') as handle:
               self.human_guidelines = pickle.load(handle)
               if len(self.human_guidelines) > self.replay_memory_size:
                   self.human_guidelines=self.human_guidelines[:self.human_guidelines]

    def set_target_network(self):
        self.target_network.load_state_dict(self.model.state_dict())
        self.target_network.eval()

    def act(self, state,exploration_rate):
        if exploration_rate < random.random():
            with torch.no_grad():
                actions = self.model(state)
                action = actions.argmax(dim=-1  )
                print('action:',action)
                return action
        else:
            action = random.randrange(self.model.actions_space_size)
            print('random action:', action)
            return torch.tensor([action]).to(self.device)

    def remember_state(self, state):
        if self.get_memory_size()< self.replay_memory_size:
            self.replay_memory.append(state)
        else:  # replace old memory with new one
            self.replay_memory.pop(0)
            self.replay_memory.append(state)
        self.memory_upload_counter+=1

    def get_memory_size(self):
        return len(self.human_guidelines) + len(self.replay_memory)

    def remember_human_guidlines(self,state):
        if self.get_memory_size() < self.replay_memory_size:
            self.human_guidelines.append(state)
        else:  # replace old memory with new one
            self.human_guidelines.pop(0)
            self.human_guidelines.append(state)
        self.memory_human_upload_counter += 1

    def sample_memory(self, batch_size,type='bot'):
        if type=="bot":
            return random.sample(self.replay_memory, batch_size)
        else:
            return random.sample(self.human_guidelines, batch_size)

    def can_sample(self, batch_size,memory_type="bot"):
        if memory_type == "bot":
            return len(self.replay_memory) >= batch_size
        if memory_type == "human":
            return len(self.human_guidelines) >= batch_size

    def save_model(self):
        torch.save(self.model.state_dict(), self.model.name+".h5")

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model.name+".h5"))

