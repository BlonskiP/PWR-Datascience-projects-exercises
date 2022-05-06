from src.Players.MCTS_tree_player import mcts_tree_player
from src.Players.Random_Player import  random_player
from src.Players.Greedy_Player import greedy_player
from src.Players.MCTS_scan_player_multi_processing import MCTS_scan_player_multi_processing
from tqdm.notebook import trange
from src.Games.Chess import no_display, play
from IPython.display import Code, display
import numpy as np
import matplotlib.pyplot as plt
import chess
import time
import glob, os
import pickle
def Research(number_of_games,player1,player2,interactive,title=None):
    print('started')
    Games = []
    range_fun = [trange,range][display==no_display] #use trange when in jupyter display mode
    start_time = time.time()
    for game in range_fun(number_of_games):
        game_temp = play(chess.Board(), player1=player1, player2=player2, interactive=interactive, max_steps=100, clear=False, display=no_display)
        try:
            Games = pickle.load(open(title + '.p', "rb"))
        except:
            pass
        Games.append(game_temp)
        pickle.dump(Games, open(title + '.p', 'wb'))
    print('Total time:',time.time()-start_time)
    plot_Game_results(Games, title)

def plot_Game_results(Games, title):
    x = []
    White = 0
    Black = 0
    Draw = 0
    time = 0
    for game in Games:
        White += game["white_wins"]
        Black += game["black_wins"]
        if game["white_wins"]==0 and game["black_wins"] ==0:
            Draw +=1
        time += game["time"]
    time = time/len(Games)
    print('Avrage one game time:',time)
    x.append(White)
    x.append(Black)
    x.append(Draw)
    objects = ('White', 'Black', 'Draw')
    y_pos = np.arange(len(objects))
    plt.bar(y_pos,x)
    plt.xlabel('Game result')
    plt.xticks(y_pos, objects)
    plt.ylabel('How Game result accrued')
    plt.title(f'Games results: {title}')
    plt.show()
def run_reasearch():
        print('Research begins')
        number_of_games=100
        #MCTS_scan vs greedy
        player1 = cust_player=(lambda b : MCTS_scan_player_multi_processing(b, scans=3, max_depth=20))
        player2 = greedy_player
        Research(number_of_games,player1,player2,interactive=False,title="MCTS scan 3")
        player1 = cust_player = (lambda b: MCTS_scan_player_multi_processing(b, scans=5, max_depth=20))
        Research(number_of_games, cust_player, player2, interactive=False, title="MCTS scan 5")
        player1 = cust_player = (lambda b: MCTS_scan_player_multi_processing(b, scans=8, max_depth=20))
        Research(number_of_games, cust_player, player2, interactive=False, title="MCTS scan 8")
        player1 = cust_player = (lambda b: mcts_tree_player(b, move_time_limit=3))
        #Research(number_of_games, player1=cust_player, player2=player2, interactive=False, title="MCTS TREE time: 3")
        player1 = cust_player = (lambda b: mcts_tree_player(b, move_time_limit=5))
        #Research(number_of_games, player1=cust_player, player2=player2, interactive=False, title="MCTS TREE time: 5")
        player1 = cust_player = (lambda b: mcts_tree_player(b, move_time_limit=8))
        #Research(number_of_games, player1=cust_player, player2=player2, interactive=False, title="MCTS TREE time: 8")
        print("Gracz Skanujący vs Gracz Tree")
        player1 = cust_player = (lambda b: MCTS_scan_player_multi_processing(b, scans=3, max_depth=20))
        player2 = cust_player2 = (lambda b: mcts_tree_player(b, move_time_limit=1))
        Research(number_of_games, player1=cust_player, player2=cust_player2, interactive=False, title="MCTS SCAN 1vs Tree3")

def plot_Pickles():
    import glob, os
    os.chdir("..")
    for file in glob.glob("*.p"):
        data = pickle.load(open(file, "rb"))
        plot_Game_results(data,file)
    pass

if __name__ == '__main__':
    print('Main starts')
    #run_reasearch()
    plot_Pickles()
else:
    'not main'
