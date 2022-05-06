from src.Players.Random_Player import random_player
import time
from tqdm.notebook import trange
from IPython.core.display import clear_output, display
import chess
"""Funkcja do losowego rozgrywania gry"""
def play(board, max_steps=100,player1=random_player, player2=random_player, interactive=False, summary=True, display=display, clear=True):
    start_time= time.perf_counter()
    players = [player1, player2]
    range_fun = [trange,range][display==no_display] #use trange when in jupyter display mode
    statuses = []
    display(board)
    for turn in range_fun(max_steps):
        if board.is_game_over():
            display("Wyjście po ruchu {}".format(turn))
            break
        for idx in range(len(players)):
            if board.is_game_over():
                break
            player = players[idx]
            (move, status) = player(board)
            board.push(move)
            if (len(status) > 0):
                statuses.append(status)
            if interactive == True:
                if clear==True:
                    clear_output()
                display("Turn {}, Player {}: {}".format(turn, idx, move))
            if interactive == True:
                display(board)
                time.sleep(0.1)
                if clear == True:
                    print('\n'.join(statuses))
    display("Gra zakończona: {}".format(board.is_game_over()))
    end_time = time.perf_counter() - start_time
    if board.is_checkmate():
        winner = {chess.WHITE: 'Czarne', chess.BLACK: 'Białe'}[board.turn] #przegrany ten, którego następna tura
        display("Zwycięzca: {}".format(winner))
    else:
        display("Brak zwycięzcy.")
    display(board)
    if not board.is_checkmate():
        display("Stalemate: {}".format(board.is_stalemate()))
        display("Insufficient_material: {}".format(board.is_insufficient_material()))
        display("Seventyfive_moves rule: {}".format(board.is_seventyfive_moves()))
        display("Fivefold_repetition: {}".format(board.is_fivefold_repetition()))
        display("Variant end condition: {}".format(board.is_variant_end()))
    if board.is_checkmate():
        display('time of game {}'.format(end_time))
        result = {chess.WHITE: [0, 1], chess.BLACK: [1, 0]}[board.turn] #przegrany ten, którego następna tura
    else:
        display('time of game {}'.format(end_time))
        result = [0, 0]
    result_dict = {'white_wins' : result[1]
                   , 'black_wins' : result[0]
                   , 'plays' : 1
                   , 'time' : end_time
                  }
    return result_dict

def no_display(x):
    pass