
#%%
import matplotlib.pyplot as plt
from numpy import log, exp, linspace
from scipy.optimize import curve_fit

nodes = [16, 32, 64, 128, 256]
iters = [17, 70, 115, 187, 572]

def fit_func(x, *args):
    a, b = args
    P = a * exp(-b * x)
    return log(0.01) / log(1 - P)

parameter_guesses = [0.05, 0.01]
popt, pcov = curve_fit(fit_func, nodes, iters, p0=parameter_guesses)

plt.plot(nodes, iters, 'o', c='black', label='data')

X = linspace(0, 300, 100)
Y = fit_func(X, *popt)
plt.plot(X, Y, label=f'fit: a={popt[0]:.2f}, b={popt[1]:.2f}', color='red', linestyle='--', linewidth=2)

plt.text(0, 700, r'$T=\frac{\ln(0.01)}{\ln(1-a \exp(-bx))}$')

plt.xlabel('Number of nodes')
plt.ylabel('Iterations to first ground state')
plt.title('Number of iterations vs. number of nodes')
plt.legend()
plt.show()
 

# %%
