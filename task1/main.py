from math import sin, cos
from numpy import arange, vectorize
from matplotlib import pyplot as plt
import json
import os

# const
A = 1.25313
x_start = -100.0
x_end = 100.0
x_step = 0.01


# our function
def f(x):
    return 0.5 + (cos(sin(x ** 2 - A ** 2)) ** 2 - 0.5) / (1 + 0.001 * (x ** 2 + A ** 2))


# create arrays
x_arrays = arange(x_start, x_end, x_step)
f2 = vectorize(f)
y_arrays = f2(x_arrays)

# make plot
plt.plot(x_arrays, y_arrays)
plt.show()

# check dir 'result'
os.mkdir('result') if not os.path.isdir('result') else print('Уже есть такая директрория')

# add to file
with open('result/data.json', 'w') as file:
    result_dir = {'data': []}
    [result_dir['data'].append({'x': x, 'y': y}) for x, y in zip(x_arrays, y_arrays)]
    file.write(json.dumps(result_dir, indent=4))
