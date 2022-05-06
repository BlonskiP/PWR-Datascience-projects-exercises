import chess
import random
import numpy as np
import time
from src.Games.Chess import play, no_display
from IPython.display import Code, display
def scan_playout(legal_moves, state, scans=10, max_depth=20):
    pass

def random_select_playout(legal_moves, state=None):
    selected_move = random.choice(legal_moves)
    return (selected_move, state)

def weighted_select_playout(legal_moves, tree):
    scores = score(legal_moves, tree)
    scores_np = np.array(scores)
    scores_norm = scores_np / scores_np.sum()
    sample = np.random.multinomial(1, scores_norm)
    idx = np.where(sample==1)[0][0]
    selected_move = legal_moves[idx]
    return (selected_move, None)

def score(legal_moves, tree):
    scores = [0] * len(legal_moves)
    for idx, move in enumerate(legal_moves):
        move = legal_moves[idx]
        if move.uci() in tree:
            summary = tree[move.uci()]
            scores[idx] = 1 + 10 * summary['white_wins']
        else:
            scores[idx] = 1
    return scores

def join_result(node, move, result):
    key = move.uci()
    if not (key in node):
        node[key] = result
        node[key]['move'] = move
    else:
        curr = node[key]
        for attr in ['white_wins', 'black_wins', 'plays']:
            curr[attr] += result[attr]
    return node

def select_best(node):
    best_score = -1
    best_move = None
    for k, v in node.items():
        if v['white_wins'] > best_score:
            best_move = v['move']
            best_score = v['white_wins']
    return best_move

def mcts_tree_player(board, strategy=random_select_playout, display=no_display, move_time_limit=3):
    color = board.turn and chess.WHITE
    legal_moves = list(board.legal_moves)

    root = {}
    
    #playout
    #number_of_plays = 200
    plays_counter = 0
    start_time = time.perf_counter()
    actual_time = 0
    while actual_time< move_time_limit :
        plays_counter += 1
        #print(actual_time)
        (playout_next_move, _) = strategy(legal_moves, root)
        board_copy = board.copy()
        board_copy.push(playout_next_move)
        result = play(board_copy, max_steps=20, display=no_display, interactive=False, summary=False)
        join_result(root, playout_next_move, result)
        new_time = time.perf_counter()
        actual_time = new_time-start_time
    
    #state
    #display(root)
    
    #selection
    selected = select_best(root)
    status = "Space {}, plays {}, selected move: {} with state {}".format(len(legal_moves), plays_counter, selected, root[selected.uci()])
    
    display(status)
    
    return (selected, status);