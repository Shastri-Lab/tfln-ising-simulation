import matplotlib.pyplot as plt
from numpy import exp, linspace, sqrt, diag, log, array
from scipy.optimize import curve_fit
import sympy as sp

nodes = array([16, 32, 64, 128, 256])
iters = array([17, 70, 115, 187, 572])

Fs = 256e9 # SPS
Fb = 106e9 # Baud
T_DSP = 24.47e-3 # seconds
N_symb = 65636 # symbols

def annealing_time(nodes, factor=1000):
    return factor * Fs/Fb * (1/Fb + T_DSP / N_symb) * nodes * nodes

def fit_func(x, *args):
    a, b, c = args
    P = a * exp(-b * x) + c
    return annealing_time(x) * log(0.01) / log(1 - P)

# using sympy for partial derivatives
a, b, c, x = sp.symbols('a b c x')
fit_func_sympy = annealing_time(x) * sp.log(0.01) / sp.log(1 - a * sp.exp(-b * x) - c)
partial_a = fit_func_sympy.diff(a)
partial_b = fit_func_sympy.diff(b)
partial_c = fit_func_sympy.diff(c)
# partial_d = fit_func_sympy.diff(d)
partial_a_func = sp.lambdify((a, b, c, x), partial_a, 'numpy')
partial_b_func = sp.lambdify((a, b, c, x), partial_b, 'numpy')
partial_c_func = sp.lambdify((a, b, c, x), partial_c, 'numpy')
# partial_d_func = sp.lambdify((a, b, d, x), partial_d, 'numpy')

# initial parameters from previous fitting
best_pars = [1.0, 0.0000001, 0.0]

solution_times = iters * annealing_time(nodes, factor=1)
popt, pcov = curve_fit(fit_func, nodes, solution_times, p0=best_pars)
perr = sqrt(diag(pcov)) 

plt.figure(dpi=120)
plt.plot(nodes, solution_times, 'o', ms=2, c='black', label='data')

X = linspace(0.0, max(nodes), 500)
Y = fit_func(X, *popt)
plt.plot(X, Y, label='fit', c='b', linestyle='-', linewidth=1)

def fit_func_error(x, params, errors):
    a, b, c = params
    delta_a, delta_b, delta_c = errors
    dy_da = partial_a_func(a, b, c, x)
    dy_db = partial_b_func(a, b, c, x)
    dy_dc = partial_c_func(a, b, c, x)
    # dy_dd = partial_d_func(a, b, d, x)
    return sqrt(
        (dy_da * delta_a)**2 + 
        (dy_dc * delta_c)**2 +
        (dy_db * delta_b)**2
        # (dy_dd * delta_d)**2 
        )

Y_error = fit_func_error(X, popt, perr)
# plt.fill_between(X, Y - Y_error, Y + Y_error, color='b', alpha=0.2, label='confidence interval')

# print a fit report
print('Fit parameters:', popt)
print('Parameter errors:', perr)
print('Covariance matrix:')
print(pcov)
print('Correlation matrix:')
print(pcov / perr / perr)

plt.xlabel('Number of nodes')
plt.ylabel('Time to Solution')
plt.yscale('log')
plt.legend()
plt.show()