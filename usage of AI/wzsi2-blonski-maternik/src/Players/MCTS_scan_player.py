
def mcts_scan_player(board, scans=10, max_depth=20):
    color = board.turn and chess.WHITE
    legal_moves = list(board.legal_moves)
    wins = [0] * len(legal_moves)
    #Definie new funcion
    def mcts_move_search(move):
    #Checks mcts move and then makes a random game. Returns who won '''
        result_for_idx = 0
        for _ in range(scans):
            board_copy = board.copy()
            board_copy.push(move)
            result_dict = play(board_copy, max_steps=20, display=no_display, interactive=False, summary=False)
            #print('thread result ',result)
            result_for_idx += result_dict[{chess.WHITE : 'white_wins', chess.BLACK : 'black_wins'}[color]] 
        #print('End thread result',result_for_idx)
        return result_for_idx
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        #get result with multithread
        results = executor.map(mcts_move_search, legal_moves)
        temp_ind = 0
        for result in results:
            wins[temp_ind] += result
            temp_ind += 1   
    display(wins)
    
    best = 0
    best_idx = random.randrange(0, len(legal_moves)) #default
    for idx in range(len(legal_moves)):
        if wins[idx] > best:
            best_idx = idx
            best = wins[idx]
    selected = legal_moves[best_idx]
    status = "Selected move: {} at idx {} from assesment {}".format(selected, best_idx, wins)
    #display(status)
    return (selected, status);