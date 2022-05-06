def human_player(board):
    text = input("Twój ruch: ")
    move = chess.Move.from_uci(text)
    while move not in board.legal_moves:
        display("Ruch {} jest niedozwolony.".format(text))
        text = input("Twój ruch: ")
        move = chess.Move.from_uci(text)
    return (move, '')