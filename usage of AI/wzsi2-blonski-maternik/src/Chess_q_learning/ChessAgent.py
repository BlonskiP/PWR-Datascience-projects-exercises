import random
import torch

from src.Chess_q_learning.ChessEnv import ChessEnv


class ChessAgent:
    def __init__(self, env: ChessEnv, policy_net=None, strategy=None):
        self.current_step = 0
        self.strategy = strategy
        self.env = env
        self.policy_net = policy_net
        self.rate = 0
    def select_action_trening(self, state, policy_net=None):
        self.current_step += 1
        exp_rate = self.strategy.get_exploration_rate(self.current_step)
        self.rate = exp_rate
        if exp_rate > random.random():
            # powinien z enva siorbać akcje i random z siorbniętych #TODO
            action = random.randrange(self.env.num_actions())
        else:
            with torch.no_grad():  # bez uczenia (wnioskowanie tylko)
                un_sq_state = state.unsqueeze(0)
                if self.policy_net is None:
                    action_nn = int(policy_net(un_sq_state).item())
                    action = self.interprete(action_nn, list(self.env.board.legal_moves))

                else:
                    action_nn = int(self.policy_net(un_sq_state).item())
                    action = self.interprete(action_nn,list(self.env.board.legal_moves))

        return action

    def select_action_game(self, state):
        self.policy_net.eval()
        un_sq_state = state.unsqueeze(0)
        action_nn = int(self.policy_net(un_sq_state).item())
        action = self.interprete(action_nn, list(self.env.board.legal_moves))
        return action

    def interprete(self,nn_action,move_list):
        inx_list = list(range(len(move_list)))
        index = min(inx_list, key=lambda x: abs(x - nn_action))
        return index