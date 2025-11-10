import matplotlib.pyplot as plt
from config import (
    PLOT_TITLE, PLOT_XLABEL, PLOT_YLABEL_SCORE, PLOT_YLABEL_EPSILON,
    COLOR_SCORE, COLOR_MEAN, COLOR_EPSILON, LINESTYLE_EPSILON,
    EPSILON_YMAX, PLOT_PAUSE
)

plt.ion()

fig = None
ax1 = None
ax2 = None

def plot(scores, mean_scores, epsilon_values):
    global fig, ax1, ax2

    if not scores or not mean_scores or not epsilon_values:
        return

    if fig is None or ax1 is None or ax2 is None:
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

    ax1.clear()
    ax2.clear()

    ax1.set_title(PLOT_TITLE)
    ax1.set_xlabel(PLOT_XLABEL)
    ax1.set_ylabel(PLOT_YLABEL_SCORE)
    ax1.plot(scores, color=COLOR_SCORE, label='Score')
    ax1.plot(mean_scores, color=COLOR_MEAN, label='Mean Score')
    ax1.set_ylim(ymin=0)
    ax1.text(len(scores)-1, scores[-1], str(scores[-1]))
    ax1.text(len(mean_scores)-1, mean_scores[-1], str(round(mean_scores[-1], 1)))

    ax2.set_ylabel(PLOT_YLABEL_EPSILON)
    ax2.plot(epsilon_values, color=COLOR_EPSILON, linestyle=LINESTYLE_EPSILON, label='Epsilon')
    ax2.set_ylim(0, max(epsilon_values[0], EPSILON_YMAX))

    fig.legend(loc='upper left')
    plt.draw()
    plt.pause(PLOT_PAUSE)