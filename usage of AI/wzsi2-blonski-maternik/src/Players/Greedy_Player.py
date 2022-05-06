import random
def greedy_player(board):
    legal_moves = list(board.legal_moves)
    drops = []
    for idx in range(len(legal_moves)):
        move = legal_moves[idx]
        if board.piece_at(move.to_square):
            drops.append(move)
        if board.is_into_check(move):
            return (move, '')
    if len(drops) > 0:
        move = random.choice(drops)
    else:
        move = random.choice(legal_moves)
    return (move, '')