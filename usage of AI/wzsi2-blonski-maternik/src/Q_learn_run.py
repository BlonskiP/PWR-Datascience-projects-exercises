from src.solution_search import decide

'Train_agent - trains q-learn agent on specific board. Returns trained agent'


class Board(object):
    def __init__(self, board):
        self.prime_board = list.copy(board)
        self.episode_board = list.copy(board)
        self.piles_num = len(board)
        self.episode_counter = 0


    def check_win(self):
        res = False
        if sum(self.episode_board) == 0:
            res = False
        elif sum(self.episode_board) == 1:
            res = True
        return res

    def is_end(self):
        res = False
        if sum(self.episode_board) <= 1:
            res = True
        return res

    def make_move(self,move):
        self.update(move[1],move[0])
        reward = 0
        win = self.check_win()
        if win:
            reward = 1
        return win, reward

    def update(self, piles, num):
        self.episode_board[piles] -= num

    def computerUpdate(self):
        self.episode_board = decide(self.episode_board, -float('inf'), float('inf'), True)[1][1]
        return self.check_win()

    # codes states as string
    def actual_state(self):
        return self.episode_board, str(self.episode_board)  # example '[1 , 2]'

    def next_possible_states(self):
        res = []
        for piles in range(self.piles_num):
            for move in range(1, self.episode_board[piles] + 1):
                temp = list(self.episode_board[:])
                temp[piles] -= move
                res.append(temp)
        return res

    def next_possible_moves(self, state ):
        res = []
        piles_count = len(state)
        for piles in range(piles_count):
            #print("piles test")
            for move in range(1, min(3, state[piles]) + 1):
             #   print("moves test")
                temp = list(state[:])
                temp[piles] -= move
                res.append([move,piles])
        return res

    def start_game(self):
        self.episode_board=list.copy(self.prime_board)
        self.episode_counter+=1
        ##print("Episode ", self.episode_board)

    def end_game(self):
        #print("Episode ",self.episode_board," ended")
        pass
    def enemy_move(self):
        return self.computerUpdate()


