from src.Chess_q_learning.player_loader import Player_loader as loader
import src.Chess_q_learning.EpsilonGreedyStrategy as epsilonStr
from src.Chess_q_learning.ChessEnv import ChessEnv


class AgentPlayer(object):
    def __init__(self, env: ChessEnv,
                 PATH='D:\Studia-DAN-D\WZSI\wzsi2-blonski-maternik\src\Chess_q_learning\chessLearned50000.99.pt'):
        self.env = env
        load = loader()
        self.agent = load.load_NN(PATH, env)

    def agent_player(self, board):
        self.env.set_board(board)
        legal_moves = list(board.legal_moves)
        state = self.env.state()
        action = self.agent.select_action_game(state)
        move = legal_moves[action]
        return (move, '')
