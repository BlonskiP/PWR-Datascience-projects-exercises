import random
import sys
sys.path.append("../../")
from statistics import mean
from torch import optim
from src.Chess_q_learning.Experience import Experience
from src.Chess_q_learning.EpsilonGreedyStrategy import EpsilonGreedyStrategy
from src.Chess_q_learning.ChessAgent import ChessAgent
from src.Chess_q_learning.ReplayMemory import ReplayMemory
from src.Chess_q_learning.DNeuralNetwork import DNeuralNetwork
from src.Chess_q_learning.Q_values_calculator import QValues
from src.Chess_q_learning.ChessEnv import ChessEnv
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import os
from tqdm.notebook import trange
from tqdm import tqdm
import torch.nn as nn
def random_player(board):
    legal_moves = list(board.legal_moves)
    move = random.choice(legal_moves)
    return (move, '')

def extract_tensors(experiences):
    batch = Experience(*zip(*experiences))
    #print('action',batch.action)
    #print('reward',batch.reward)
    state_tensor = torch.stack(batch.state)
    action_tensor = torch.stack(batch.action)
    reward_tensor = torch.stack(batch.reward)
    next_state_tensor = torch.stack(batch.next_state)

    return state_tensor, action_tensor, reward_tensor, next_state_tensor
loses = 0
wins = 0
draws = 0
batch_size = 512
gamma = 0.8
#gamma = 0.1 #which is the discount factor used in the Bellman equation
epsilon_start = 1
epsilon_end = 0.01
epsilon_decay = 0.001
target_update = 10
memory_size = 5000
learning_rate = 0.01
num_episodes = 1000  # Ile gier
steps_per_episode = 40
model_name = str(num_episodes)+str(gamma)
PATH = os.path.dirname(os.path.realpath(__file__))+'\\chessLearned'+model_name
# env = Chess_evn_manager() #TODO
env = ChessEnv()
enemy = random_player
strategy = EpsilonGreedyStrategy(epsilon_start, epsilon_end, epsilon_decay)  # Corrent stratedy
chess_agent = ChessAgent(strategy=strategy, env=env)
memory = ReplayMemory(memory_size)

# DNN's
policy_net = DNeuralNetwork()
target_net = DNeuralNetwork()
target_net.load_state_dict(policy_net.state_dict())
target_net.eval() #Evalue mode - not learning mode

optimizer = optim.Adam(params=policy_net.parameters(), lr=learning_rate)
num_ep = 0
cum_rewards_per_episode = []
cum_reward = 0
reward_per_episode = []
losses = []
losses_plot = []
for episode in tqdm(range(num_episodes)):
    # reset env here
    env.reset()
    num_ep += 1
    #print(num_ep)
    episode_reward = []
    rewards_per_episode_temp = 0
    for i in range(steps_per_episode):
       # print('step ',i, ' episode', num_ep)
        # state = get state from env as 'state'
        if(enemy == None):
            reverse_state = env.turnBlack()    #reverse state while learning
            state = env.state(reverse_state)
        else:
            state = env.state()
        action = chess_agent.select_action_trening(state, policy_net)
       # print(f"{ list(env.board.legal_moves)[action]} nr {action} z {env.board.legal_moves} " )
        # reward = wykonaj ruch z enva i dostań reward jakiś
        reward, done, won = env.step(action)
        reward -= 1/steps_per_episode
        if done==True and won==False: #punish for draw
            reward = -1
        cum_reward += reward
        rewards_per_episode_temp += reward
        # next_state =   get new state from env
        next_state = env.state()
        memory.push(Experience(state, torch.tensor(action), next_state, torch.tensor(reward).float()))
        # state = next_state update states
        # ???

        # uczenie po napełnieniu pamieci
        if memory.can_provide_sample(batch_size):
            experiences = memory.sample(batch_size)
            states, actions, rewards, next_states = extract_tensors(experiences)
         #   print('trening',states.shape)
            current_Q_values = QValues.get_current(policy_net, states, actions)
            next_Q_values = QValues.get_next(target_net, next_states)
            target_Q_values = (next_Q_values * gamma) + rewards

            loss = F.smooth_l1_loss(current_Q_values, target_Q_values.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss)
            losses_plot.append(loss/len(losses))






        # if env.is_done() sprawdz czy gra sie skonczyła dokonaj jakiś pomiarów itp no i przerwij episod

        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
        if won:
            wins+=1
            #print(str(num_ep),str(i),done, 'win',rewards_per_episode_temp,cum_reward)
            break
        if done:
            draws += 1  # jezeli skonczyły sie ruchy
            #print(str(num_ep), str(i), done, 'draw', rewards_per_episode_temp, cum_reward)
            break
        if(enemy != None):
            (move, status) = enemy(env.board)
            env.board.push(move)
            done = env.board.is_game_over()
        if done:
            loses += 1
            #print(str(num_ep), str(i), done, 'lose',rewards_per_episode_temp,cum_reward)
            break
    else:
        draws += 1  # jezeli skonczyły sie ruchy
        #print(str(num_ep), str(i), done, 'draw', rewards_per_episode_temp, cum_reward)



    reward_per_episode.append(rewards_per_episode_temp)
    cum_rewards_per_episode.append(cum_reward)
# END TRENING
print(chess_agent.rate,'epi',episode)
print('wins',wins)
print('draws',draws)
print('loses',loses)

torch.save(target_net,PATH+'.pt')
# PLOT RESULTS ect
plt.plot(cum_rewards_per_episode)
plt.title("Cum rewards"+model_name)
plt.savefig('cumrewrds'+model_name+'.png')
plt.show()

plt.title("Rewards per episode"+model_name)
plt.plot(reward_per_episode)
plt.savefig('rewardsPerEpisode'+model_name+'.png')
plt.show()


plt.title("loss"+model_name)
plt.plot(losses)
plt.savefig('loss'+model_name+'.png')
plt.show()

plt.title("loss"+model_name)
plt.plot(losses_plot)
plt.savefig('loss_mean'+model_name+'.png')
plt.show()