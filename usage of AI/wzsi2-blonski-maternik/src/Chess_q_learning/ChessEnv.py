import chess
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
import torch


class ChessEnv:
    def __init__(self, state=None, board=None):
        if board is None:
            self.board = chess.Board()
        else:
            self.board = board
        self.label_encoder = LabelEncoder()
        self.onehot_encoder = OneHotEncoder(sparse=False)
        self.init_encoders()

    def set_board(self,board):
        self.board = board

    def init_encoders(self):
        values = array(self.preprocess_state())
        self.label_encoder.fit(values)
        state_encoded = self.label_encoder.transform(values)
        state_encoded = state_encoded.reshape(len(state_encoded), 1)
        self.onehot_encoder.fit(state_encoded)

    def step(self, action):
        move = list(self.board.legal_moves)[action]
        #print('action:',action,'\n')
        #print('moves.count',self.board.legal_moves.count(),'\n')
        #print('moves',list(self.board.legal_moves),'\n')
        #print('move: ',move,'\n')
        reward = 0
        won = False
        if self.board.is_capture(move): #czy bicie
            reward += 0.25

        self.board.push(move)

        if self.board.is_check(): #czy check
            reward += 0.5

        if self.board.is_checkmate(): #czy szach-mat
            reward += 1
            won = True

        done = self.board.is_game_over() #czy wygrana
        return reward, done, won

    def durnWrite(self):
        return self.board.turn == chess.WRITE

    def turnBlack(self):
        return self.board.turn == chess.BLACK

    def state(self, reverse_state=False):
        state = self.preprocess_state(reverse_state)
        state = self.onehot(state)
        return state

    def num_actions(self):
        return self.board.legal_moves.count()

    def reset(self):
        self.board.reset()

    def preprocess_state(self, reversed_state=False):
        state = str(self.board).replace("\n", " ").split()
        return state

    def onehot(self, state):
        values = array(state)
        state_encoded = self.label_encoder.transform(values)
        state_encoded = state_encoded.reshape(len(state_encoded), 1)
        onehot_encoded = self.onehot_encoder.transform(state_encoded)
        # if reversed_state:
        #    state = ''.join(c.lower() if c.isupper() else c.upper() for c in state)
        onehot_tensor = torch.from_numpy(onehot_encoded)
        return onehot_tensor
