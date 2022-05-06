from src.Chess_q_learning.ChessAgent import ChessAgent
from src.Chess_q_learning.DNeuralNetwork import DNeuralNetwork
import torch

class Player_loader:
    def load_NN(self,PATH,env):
        model = DNeuralNetwork()
        model = torch.load(PATH)
        model.eval()
        self.nn_player = ChessAgent(env, policy_net=model)
        return self.nn_player
