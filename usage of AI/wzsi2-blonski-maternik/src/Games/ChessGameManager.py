

class ChessManager(GameManager):

    def load_game_state(self, state):
        self.board = chess.Board(state)


    def save_game_state(self):
        self.saved_game_states.append(self.board.fen())
        return self.saved_game_states[-1] #last item form list

    def __init__(self):
        super()

    def init_game(self):
        self.board = chess.Board()
        self.Player_1 = StupidPlayer()  # for test only
        self.Player_2 = StupidPlayer()  # for test only
        self.actual_player = self.Player_1
        self.saved_game_states = []
        pass

    def game_actions(self):  # Called in game loop
        print(self.board)
        moves_list = self.possible_moves()
        print(moves_list)
        move_choosen = self.actual_player.make_move(moves_list)
        self.board.push(move_choosen)

        self.change_player()
        self.is_playing = not self.check_win()

    def check_win(self):
        return self.board.is_game_over(claim_draw = self.board.can_claim_fifty_moves())  # i hope its ok

    def change_player(self):
        if self.actual_player == self.Player_1:
            self.actual_player = self.Player_2
        else:
            self.actual_player = self.Player_1

    def possible_moves(self):
        moves_list = list(self.board.legal_moves)
        return moves_list