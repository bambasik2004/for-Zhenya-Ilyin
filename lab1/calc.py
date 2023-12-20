from math import sqrt
import inspect
from numpy import round
from scipy import constants

# const
c = constants.c
pi = constants.pi
mu_0 = constants.mu_0
e_0 = constants.epsilon_0
f = 25 * 10 ** 9
e_2 = 2.56
w_0 = 120 * pi


def e_print(*args):
    callers_local_args = inspect.currentframe().f_back.f_locals.items()
    for arg_name, arg_val in callers_local_args:
        if (arg_val in args) and (0.009 < abs(arg_val) < 1000):
            print(f'{arg_name}: {round(arg_val, 3)}')
        elif arg_val in args:
            print(f'{arg_name}: {arg_val:.2e}')


# calc
T = 1 / f
e_print(T)
speed = c/sqrt(e_2)
e_print(speed)
wave_len_0 = c / f
e_print(wave_len_0)
wave_len = speed / f
e_print(wave_len)
w = w_0 / sqrt(e_2)
e_print(w)
G = (w - w_0) / (w + w_0)
e_print(G)
T = (2 * w) / (w + w_0)
e_print(T)
L = 10 * wave_len_0
e_print(L)
x_l = L / 2
e_print(x_l)
x_i = L / 4
e_print(x_i)
x_1 = L / 8
e_print(x_1)
x_2 = x_i
e_print(x_2)
x_3 = 3 * L / 4
e_print(x_3)
t_max = L / speed
e_print(t_max)

# table 2
N_values = '''100
90
80
70
60
50
40
30
25
20
18
16
14
12
10
9
8
7
6
5
'''
N_range = list(map(int, N_values.split('\n')[:-1]))
D_range = [wave_len_0 / N for N in N_range]
[print(round(D, 6)) for D in D_range]
print(D_range)