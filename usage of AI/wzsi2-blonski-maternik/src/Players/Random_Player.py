import random
def random_player(board):
    legal_moves = list(board.legal_moves)
    move = random.choice(legal_moves)
    return (move, '')