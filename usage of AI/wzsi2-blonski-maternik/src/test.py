import src.Q_learn_run as board
import matplotlib.pyplot as plt

from src.nim_q_learn import Q_learning_player

progress = []
env = board.Board([1, 1])
Q_learing = Q_learning_player(env)
for i in range(1,20,10):

    history = Q_learing.train(i, 100)
    progress.append(history['cum_rewards'][-1]/i)

plt.plot(progress)

Q_learing.exploration_rate = Q_learing.min_exploration_rate
for i in range(1,40,10):
    env = board.Board([10, 10])
    history = Q_learing.train(i, 100)
    progress.append(history['cum_rewards'][-1]/i)
plt.plot(progress)