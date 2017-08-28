"""Script comparing quality and computing time for spike inference using OASIS,
an active set method for sparse nonnegative deconvolution
Created on Thu Aug 4
@author: Johannes Friedrich
"""

import numpy as np
from timeit import Timer
from oasis import oasisAR1, constrained_oasisAR1
from oasis.functions import gen_sinusoidal_data, estimate_parameters


g = .95
sn = .3
Y, trueC, trueSpikes = gen_sinusoidal_data(b=10)
N, T = Y.shape

results = {}
for opt in ['-', 'l', 'lb', 'lbg', 'lbg10', 'lbg5', 'lbg_ds', 'lbg10_ds', 'lbg5_ds']:
    results[opt] = {}
    results[opt]['time'] = []
    results[opt]['distance'] = []
    results[opt]['correlation'] = []
    for i, y in enumerate(Y):
        g, sn = estimate_parameters(y, p=1, fudge_factor=.99, method='logmexp')
        lam = 0
        b = np.percentile(y, 15)
        if opt == '-':
            foo = lambda y: oasisAR1(y - b, g, lam)
        elif opt == 'l':
            foo = lambda y: constrained_oasisAR1(y - b, g, sn)
        elif opt == 'lb':
            foo = lambda y: constrained_oasisAR1(y, g, sn, optimize_b=True)
        elif opt == 'lbg':
            foo = lambda y: constrained_oasisAR1(y, g, sn, optimize_b=True, optimize_g=len(y))
        elif opt == 'lbg10':
            foo = lambda y: constrained_oasisAR1(y, g, sn, optimize_b=True, optimize_g=10)
        elif opt == 'lbg5':
            foo = lambda y: constrained_oasisAR1(y, g, sn, optimize_b=True, optimize_g=5)
        elif opt == 'lbg_ds':
            foo = lambda y: constrained_oasisAR1(
                y, g, sn, optimize_b=True, optimize_g=len(y), decimate=10)
        elif opt == 'lbg10_ds':
            foo = lambda y: constrained_oasisAR1(
                y, g, sn, optimize_b=True, optimize_g=10, decimate=10)
        elif opt == 'lbg5_ds':
            foo = lambda y: constrained_oasisAR1(
                y, g, sn, optimize_b=True, optimize_g=5, decimate=10)
        results[opt]['time'].append(Timer(lambda: foo(y)).timeit(number=10))
        s = foo(y)[1]
        results[opt]['distance'].append(np.linalg.norm(s - trueSpikes[i]))
        results[opt]['correlation'].append(np.corrcoef(s, trueSpikes[i])[0, 1])

print(' optimize     Time [ms]       Distance       Correlation')
for opt in ['-', 'l', 'lb', 'lbg', 'lbg10', 'lbg5', 'lbg_ds', 'lbg10_ds', 'lbg5_ds']:
    print(('%8s   %6.2f+- %.2f   %.2f +- %.2f    %.3f +- %.3f' if opt == '-'
           else '%8s  %6.1f +- %.1f    %.2f +- %.2f    %.3f +- %.3f') %
          (opt,
           np.mean(results[opt]['time']) * 100, np.std(results[opt]['time']) * 100 / np.sqrt(N),
           np.mean(results[opt]['distance']), np.std(results[opt]['distance']) / np.sqrt(N),
           np.mean(results[opt]['correlation']), np.std(results[opt]['correlation']) / np.sqrt(N)))
