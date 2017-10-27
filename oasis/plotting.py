import matplotlib.pyplot as plt


def init_fig():
    """change some defaults for plotting"""
    plt.rc('figure', facecolor='white', dpi=90, frameon=False)
    plt.rc('font', size=30, **{'family': 'sans-serif', 'sans-serif': ['Computer Modern']})
    plt.rc('lines', lw=2)
    plt.rc('text', usetex=True)
    plt.rc('legend', **{'fontsize': 24, 'frameon': False, 'labelspacing': .3, 'handletextpad': .3})
    plt.rc('axes', linewidth=2)
    plt.rc('xtick.major', size=10, width=1.5)
    plt.rc('ytick.major', size=10, width=1.5)


def simpleaxis(ax):
    """plot only x and y axis, not a frame for subplot ax"""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
