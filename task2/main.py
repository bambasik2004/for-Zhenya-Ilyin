import numpy
import requests
import xml.etree.ElementTree as ET
from scipy import special, constants
from numpy import arange, abs, sum
from matplotlib import pyplot as plt
import json

# parse XML
data = requests.get('https://jenyay.net/uploads/Student/Modelling/task_02.xml').text
root = ET.fromstring(data)
for element in root.iter('variant'):
    if element.get('number') == '10':
        D = float(element.get('D'))
        f_min = float(element.get('fmin'))
        f_max = float(element.get('fmax'))

# const
n_end = 10
f_step = 100000
r = D / 2
f_arange = arange(f_min, f_max, f_step)
wavelength_arange = constants.c / f_arange
k_arange = 2 * constants.pi / wavelength_arange


# h
def f4(n, x):
    return special.spherical_jn(n, x) + 1j * special.spherical_yn(n, x)


# b
def f3(n, x):
    return (x * special.spherical_jn(n - 1, x) - n * special.spherical_jn(n, x)) / (x * f4(n - 1, x) - n * f4(n, x))


# a
def f2(n, x):
    return special.spherical_jn(n, x) / f4(n, x)


# Radar Cross Section(RCS) - ЭПР
rcs_arange = (wavelength_arange ** 2) / numpy.pi * (abs(sum([((-1) ** n) * (n+0.5) * (f3(n, k_arange * r) - f2(n, k_arange * r)) for n in range(1, n_end)], axis=0)) ** 2)

# File
with open('result/data.json', 'w') as file:
    result_dir = {'data': [{'freq': f, 'lamda': wavelength, 'rcs': rcs} for f, wavelength, rcs in zip(f_arange, wavelength_arange, rcs_arange)]}
    file.write(json.dumps(result_dir, indent=4))

# Graphic
plt.plot(f_arange, rcs_arange)
plt.show()

