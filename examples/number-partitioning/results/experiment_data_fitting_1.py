import matplotlib.pyplot as plt
from numpy import exp, linspace, sqrt, diag
from scipy.optimize import curve_fit
import sympy as sp

nodes = [16, 32, 64, 128, 256]
TTS = [0.0039, 0.0647, 0.4249, 2.7636, 33.8132]

def fit_func(x, a, b, c):
    return a * exp(b * x) + c

# using sympy for partial derivatives
a, b, c, x = sp.symbols('a b c x')
fit_func_sympy = a * sp.exp(b * x) + c
partial_a = fit_func_sympy.diff(a)
partial_b = fit_func_sympy.diff(b)
partial_c = fit_func_sympy.diff(c)
partial_a_func = sp.lambdify((a, b, c, x), partial_a, 'numpy')
partial_b_func = sp.lambdify((a, b, c, x), partial_b, 'numpy')
partial_c_func = sp.lambdify((a, b, c, x), partial_c, 'numpy')

# initial parameters from previous fitting
best_pars = [
    3.0005713708e-01,
    1.8509532979e-02,
    -4.6947118345e-01,
]

popt, pcov = curve_fit(fit_func, nodes, TTS, p0=best_pars)
perr = sqrt(diag(pcov)) 

plt.figure(dpi=120)
plt.plot(nodes, TTS, 'o', ms=2, c='black', label='data')

X = linspace(min(nodes), max(nodes), 100)
Y = fit_func(X, *popt)
plt.plot(X, Y, label='fit', c='b', linestyle='-', linewidth=1)

def fit_func_error(x, params, errors):
    a, b, c = params
    delta_a, delta_b, delta_c = errors
    dy_da = partial_a_func(a, b, c, x)
    dy_db = partial_b_func(a, b, c, x)
    dy_dc = partial_c_func(a, b, c, x)
    return sqrt((dy_da * delta_a) ** 2 + (dy_db * delta_b) ** 2 + (dy_dc * delta_c) ** 2)

Y_error = fit_func_error(X, popt, perr)
plt.fill_between(X, Y - Y_error, Y + Y_error, color='b', alpha=0.3, label='confidence interval')

plt.xlabel('Number of nodes')
plt.ylabel('Time to Solution')
plt.yscale('log')
plt.legend()
plt.show()