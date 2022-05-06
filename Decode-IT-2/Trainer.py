import math
import random
import time
from collections import namedtuple
from statistics import mean

import numpy as np
import torch
import torch.nn.functional as F

#torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.utils.tensorboard import SummaryWriter
from Qvalues import Qvalues

Experience = namedtuple('Experience',('state','action','next_state','reward'))
class RfTrainer:

    def __init__(
            self,
            em,
            agent,
            learning_rate=0.1,
            gamma=0.99,
            min_exploration_rate=0.01,
            max_exploration_rate=0.9,
            exploration_decay = 0.001,
            num_episodes=100,
            max_steps_per_episode=100,
            device='cpu',
            target_update=20,
            batch_size=100,
            openAI_gym=False):
        self.em = em
        self.openAi_gym=openAI_gym
        self.gamma=gamma
        self.example_final_count=0
        self.target_update=target_update
        self.batch_size=batch_size
        self.agent = agent
        self.lr = learning_rate
        self.min_exploration_rate = min_exploration_rate
        self.max_exploration_rate = max_exploration_rate
        self.exploration_decay = exploration_decay
        self.example_input_count = 0
        self.num_episodes = num_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.device = device
        self.writer = SummaryWriter(log_dir=f"logs/{agent.name}")

    def train(self):
        rewards_all_episodes = []
        step_counter =0
        steps_to_train_counter = 0
        self.example_final_count=0

        for episode in range(self.num_episodes):
            print("EPISODE", episode, "STARTED ----------------------------------")
            episode_reward = 0.0
            self.em.reset()
            state = self.em.get_state()

            #self.writer.add_scalar('exploration rate', rate, episode)
            for timestep in range(self.max_steps_per_episode):
                rate = self.get_exploration_rate(episode)
                step_counter+=1
                if self.em.env.BotController.activate:
                    action = self.agent.act(state,rate)
                else:
                    action = self.em.human_input()
                reward = self.em.take_action(action)
                episode_reward += reward
                new_state = self.em.get_state()
                if self.em.env.BotController.activate:
                    self.agent.remember_state(Experience(state,action,new_state,reward))
                else:
                    self.agent.remember_human_guidlines(Experience(state,action,new_state,reward))
                self.raport_to_tensorboard_episode(reward, step_counter, episode_reward, episode, rate,action)
                self.raport_inputs_to_tensorboard(state)
                state = new_state

                self.QTable_optim(episode)

                if self.em.done:
                    self.agent.upload_memory()
                    rewards_all_episodes.append(episode_reward)
                    self.raport_to_tensorboard_end_episode(timestep,episode,rewards_all_episodes,state)
                    self.update_target(episode)
                    print("EPISODE",episode,"ENDED ----------------------------------")
                    break #break episode
    def Qvalues(self,experiences):
        states, actions, rewards, next_states = self.extract_tensors(experiences)

        current_q_values = Qvalues.get_current(self.agent.model, states, actions)
        next_q_values = Qvalues.get_next(self.agent.target_network, next_states)

        target_q_values = (next_q_values * self.gamma) + rewards
        return current_q_values, target_q_values

    def QTable_optim(self,episode):
        sample_new_exp=False
        sample_human_guide = False
        if self.agent.can_sample(self.batch_size):
            sample_new_exp = True
        if self.agent.can_sample(self.batch_size,memory_type="human"):
            sample_human_guide= True
        batch_size = self.batch_size
        loss_sum = 0
        if sample_new_exp and sample_human_guide:
            batch_size=int(batch_size/2)
        if sample_new_exp or sample_human_guide:
            t1= time.time()
            if sample_new_exp:
                experiences_bot = self.agent.sample_memory(self.batch_size)
                current_q_values, target_q_values=self.Qvalues(experiences_bot)
                loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
                loss_sum+=loss
                self.agent.optimizer.zero_grad()
                loss.backward()
                self.agent.optimizer.step()
            if sample_human_guide:
                experiences_human = self.agent.sample_memory(self.batch_size,type="bot")
                current_q_values, target_q_values = self.Qvalues(experiences_human)
                loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
                loss_sum += loss
                self.agent.optimizer.zero_grad()
                loss.backward()
                self.agent.optimizer.step()
            self.raport_to_tensorboard_trainig(loss_sum, episode)
            print("learning time:",time.time()-t1)

    def raport_inputs_to_tensorboard(self,state):
        if self.example_input_count < 100:
            img_batch = torch.squeeze(state)
            img_batch = torch.unsqueeze(img_batch, dim=1)
            self.writer.add_images('example inputs', img_batch, self.example_input_count)
            self.example_input_count += 1

    def raport_to_tensorboard_trainig(self,loss,episode):
        self.writer.add_scalar('loss', torch.sum(loss), episode)

    def raport_to_tensorboard_episode(self,reward,step_counter,episode_reward,episode,rate,action):
        self.writer.add_scalar('reward', reward.item(), step_counter)
        self.writer.add_scalar('episode reward', episode_reward, step_counter)
        print(action.item(),"rate: ", rate, " reward", reward.item(), 'mem:', len(self.agent.replay_memory), "/",
              self.agent.replay_memory_size)


    def raport_to_tensorboard_end_episode(self,timestep,episode,rewards_all_episodes,state):
        self.writer.add_scalar('steps per episode', timestep, episode)
        self.writer.add_scalar('sum_rewards_per_episode', sum(rewards_all_episodes), episode)
        self.writer.add_scalar('mean_rewards_per_episode',
                               sum(rewards_all_episodes) / len(rewards_all_episodes), episode)
        if self.example_final_count <= 100:
            img_batch = torch.squeeze(state)
            img_batch = torch.unsqueeze(img_batch, dim=1)
            self.writer.add_images('finale_batch', img_batch, self.example_final_count)
            self.example_final_count += 1

    def get_exploration_rate(self,current_step):
        rate = self.min_exploration_rate + \
               (self.max_exploration_rate - self.min_exploration_rate) \
               * math.exp(-1*current_step*self.exploration_decay)

        return rate

    def extract_tensors(self, experiences):

        batch = Experience(*zip(*experiences))
        states = torch.cat(batch.state)
        actions = torch.cat(batch.action)
        rewards = torch.cat(batch.reward)
        next_states = torch.cat(batch.next_state)
        return (states,actions,rewards,next_states)

    def update_target(self,episode):
        if episode % self.target_update == 0:
            if not self.openAi_gym:#pause
                self.em.env.pause_env(reason="PAUSE - Coping model=target")
            self.agent.save_model()
            self.agent.set_target_network()
            if not self.openAi_gym:#unpause
                self.em.env.pause_env(reason="PAUSE END- Coping model=target")
                time.sleep(0.5)