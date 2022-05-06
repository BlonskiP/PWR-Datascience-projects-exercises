import numpy as np
import random
import ast
from tqdm.notebook import trange
from src.Q_learn_run import Board

class Q_learning_player:
    def __init__(self, Board):
        self.board = Board

        # learning params
        self.learning_rate = 0.7
        self.discount_rate = 0.99
        # exploration params
        self.min_exploration_rate = 0.01
        self.max_exploration_rate = 1
        self.exploration_rate = self.max_exploration_rate
        self.exploration_decay = 0.001
        self.episodes_tought = 0
        # For metrics and plots
        self.history = {
            "rewards": [],
            "cum_rewards": [],# updatable array of reward history per episode
            "exploration_decay":[]
        }
        #Q table - On the begining of the NIM game all posibilites
        self.Q_table =  dict()
        possible_states = self.board.next_possible_states()
        possible_states.append(self.board.actual_state()[0])
        for state in possible_states: #for each starting state make Q value = 0
            moves = self.board.next_possible_moves(state)
            self.Q_table[str(state)] = {str(move): 0 for move in moves}
        print("Q table:",self.Q_table)
        #print("Q learning player initialized. use .train(num,steps)")

    def train(self, num_episodes, max_steps):
        cum_reward = 0
        # train loop
        for episode in trange(num_episodes):
            self.episodes_tought += 1
            is_done = False
            reward_gain = 0
            enemy_won = False
            self.board.start_game()
            for step in range(max_steps):

                actual_state, actual_state_key = self.board.actual_state()
                moves = self.board.next_possible_moves(actual_state)
                check_action_type = random.uniform(0, 1)
                if check_action_type > self.exploration_rate:
                    actions_for_state = self.Q_table[actual_state_key]
                    action_key = max(actions_for_state) # get best action you know for this state
                    action =  ast.literal_eval(action_key)
                else:
                    if len(moves) <=0:
                        break;
                    action = random.choice(moves) # get random action for that state

                # GET REWARD STATE ECT FROM GAME
                # self.upadte_Q_table(self.state,new_state,action,reward)
                #
                is_won, reward = self.board.make_move(action)
                reward += -1 / max_steps
                is_end = self.board.is_end()
                new_State, new_State_key = self.board.actual_state()
                if is_end == False:
                    enemy_won = self.board.enemy_move()
                if enemy_won:
                    reward = -3
                is_end = self.board.is_end()
                if is_end == True:
                    is_won = self.board.check_win() #check if enemy lose
                    if is_won:
                        reward = 1

                self.update_keys_Q_table() # add new states and moves to Q-table
                self.update_Q_table(actual_state_key,str(action),reward,new_State,new_State_key)
                reward_gain += reward
                if is_end == True:  # break episode loop is game is finished
                    break;

            #loop end

            # EPISODE IS FINISHED
            self.update_exploration_rate(self.episodes_tought)
            self.history['rewards'].append(reward_gain)
            cum_reward+=reward_gain
            self.history['cum_rewards'].append(cum_reward)
            self.board.end_game()
        return self.history

    def update_Q_table(self, state, action, reward,new_State,new_State_key):

        #count Q value
        #print(state)
        max_value = max(self.Q_table[new_State_key].values())
        q_table_value = self.Q_table[state][action]
        new_Q_value = q_table_value * (1 - self.learning_rate) + \
                                      self.learning_rate * (
                                              reward + self.discount_rate * max_value)
        #update table
        self.Q_table[state][action] = new_Q_value
        #update State
        self.state = new_State

    def update_exploration_rate(self, episode_number):
        self.history['exploration_decay'].append(self.exploration_rate)
        self.exploration_rate = self.min_exploration_rate + (
                self.max_exploration_rate - self.min_exploration_rate) * np.exp(
            -self.exploration_decay * episode_number)

    def update_keys_Q_table(self):
        possible_states = self.board.next_possible_states()
        possible_states.append(self.board.actual_state()[0])
        for state in possible_states: #for each new state / move make Q value = 0
            moves = self.board.next_possible_moves(state)
            if str(state) not in self.Q_table: #init new state
                self.Q_table[str(state)] = {str(move): 0 for move in moves}
            else:
                dic_moves_keys = [str(move) for move in moves]
                state_dic = self.Q_table[str(state)]
                for key in dic_moves_keys:
                    if key not in state_dic:
                        state_dic[key] = 0


