"""Script illustrating the autoregressive calcium fluorescence model
for OASIS, an active set method for sparse nonnegative deconvolution
@author: Johannes Friedrich
"""

from matplotlib import pyplot as plt
from oasis.functions import gen_data
from oasis.plotting import init_fig, simpleaxis

init_fig()
# colors for colorblind from  http://www.cookbook-r.com/Graphs/Colors_(ggplot2)/
col = ['#0072B2', '#009E73', '#D55E00']


Y, trueC, trueSpikes = gen_data([1.58, -.6], .5, T=455,
                                framerate=30, firerate=2., seed=0)
plt.figure(figsize=(15, 2.5))
for i, t in enumerate(trueSpikes[0, 20:-1]):
    if t:
        plt.plot([i, i], [0, 1], c=col[2])
plt.plot([trueSpikes[0, -1], trueSpikes[0, -1]], [0, 1], c=col[2], label=r'$s$')
plt.plot(trueC[0, 20:] / 3., c=col[0], label=r'$c$', zorder=-11)
plt.scatter(range(435), Y[0, 20:] / 3., c=col[1], clip_on=False, label=r'$y$')
plt.legend(loc=(.38, .75), ncol=3)
plt.yticks([0, 1, 2], [0, 1, 2])
plt.xticks(*[[0, 150, 300]] * 2)
plt.xlim(0, 435)
plt.ylim(0, 2.9)
plt.ylabel('Fluorescence', y=.45)
plt.xlabel('Time', labelpad=-15)
simpleaxis(plt.gca())
plt.tight_layout(pad=.01)
plt.show()
