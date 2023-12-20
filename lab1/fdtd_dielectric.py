# -*- coding: utf-8 -*-
'''
Моделирование отражения гармонического сигнала от слоя диэлектрика
'''

import math
import time

import numpy as np
import numpy.typing as npt

from objects import LayerContinuous, LayerDiscrete, Probe

import boundary
import sources
import tools

import re
import calc
from numpy import cos, arcsin, sin, sqrt
import matplotlib.pyplot as plt

class Sampler:
    def __init__(self, discrete: float):
        self.discrete = discrete

    def sample(self, x: float) -> int:
        return math.floor(x / self.discrete + 0.5)


def sampleLayer(layer_cont: LayerContinuous, sampler: Sampler) -> LayerDiscrete:
    start_discrete = sampler.sample(layer_cont.xmin)
    end_discrete = (sampler.sample(layer_cont.xmax)
                    if layer_cont.xmax is not None
                    else None)
    return LayerDiscrete(start_discrete, end_discrete,
                         layer_cont.eps, layer_cont.mu, layer_cont.sigma)


def fillMedium(layer: LayerDiscrete,
               eps: npt.NDArray[np.float64],
               mu: npt.NDArray[np.float64],
               sigma: npt.NDArray[np.float64]):
    if layer.xmax is not None:
        eps[layer.xmin: layer.xmax] = layer.eps
        mu[layer.xmin: layer.xmax] = layer.mu
        sigma[layer.xmin: layer.xmax] = layer.sigma
    else:
        eps[layer.xmin:] = layer.eps
        mu[layer.xmin:] = layer.mu
        sigma[layer.xmin:] = layer.sigma


if __name__ == '__main__':
    # Делаем цикл чтобы построить табличку для всех значенйи D_x
    answer = []
    # Создадим списки для графиков
    G_fdtd_list = []
    G_volna_list = []
    T_fdtd_list = []
    T_volna_list = []
    d_g_list = []
    d_t_list = []

    for cur_D, cur_N in zip(calc.D_range, calc.N_range):
        # Используемые константы
        # Волновое сопротивление свободного пространства
        W0 = 120.0 * np.pi

        # Скорость света в вакууме
        c = 299792458.0

        # Электрическая постоянная
        eps0 = 8.854187817e-12

        # Параметры моделирования
        # Частота сигнала, Гц
        f_Hz = calc.f

        # Дискрет по пространству в м
        dx = cur_D

        wavelength = c / f_Hz
        Nl = wavelength / dx

        # Число Куранта
        Sc = 1.0

        # Размер области моделирования в м
        maxSize_m = calc.L

        # Время расчета в секундах
        maxTime_s = calc.t_max

        # Положение источника в м
        sourcePos_m = calc.x_i

        # Координаты датчиков для регистрации поля в м
        probesPos_m = [calc.x_1, calc.x_2, calc.x_3]

        # Параметры слоев
        layers_cont = [LayerContinuous(calc.x_l, eps=calc.e_2, sigma=0.0)]

        # Скорость обновления графика поля
        speed_refresh = 500

        # Переход к дискретным отсчетам
        # Дискрет по времени
        dt = dx * Sc / c

        sampler_x = Sampler(dx)
        sampler_t = Sampler(dt)

        # Время расчета в отсчетах
        maxTime = sampler_t.sample(maxTime_s)

        # Размер области моделирования в отсчетах
        maxSize = sampler_x.sample(maxSize_m)

        # Положение источника в отсчетах
        sourcePos = sampler_x.sample(sourcePos_m)

        layers = [sampleLayer(layer, sampler_x) for layer in layers_cont]

        # Датчики для регистрации поля
        probesPos = [sampler_x.sample(pos) for pos in probesPos_m]
        probes = [Probe(pos, maxTime) for pos in probesPos]

        # Вывод параметров моделирования
        print(f'Число Куранта: {Sc}')
        print(f'Размер области моделирования: {maxSize_m} м')
        print(f'Время расчета: {maxTime_s * 1e9} нс')
        print(f'Координата источника: {sourcePos_m} м')
        print(f'Частота сигнала: {f_Hz * 1e-9} ГГц')
        print(f'Длина волны: {wavelength} м')
        print(f'Количество отсчетов на длину волны (Nl): {Nl}')
        probes_m_str = ', '.join(['{:.6f}'.format(pos) for pos in probesPos_m])
        print(f'Дискрет по пространству: {dx} м')
        print(f'Дискрет по времени: {dt * 1e9} нс')
        print(f'Координаты пробников [м]: {probes_m_str}')
        print()
        print(f'Размер области моделирования: {maxSize} отсч.')
        print(f'Время расчета: {maxTime} отсч.')
        print(f'Координата источника: {sourcePos} отсч.')
        probes_str = ', '.join(['{}'.format(pos) for pos in probesPos])
        print(f'Координаты пробников [отсч.]: {probes_str}')

        # Параметры среды
        # Диэлектрическая проницаемость
        eps = np.ones(maxSize)

        # Магнитная проницаемость
        mu = np.ones(maxSize - 1)

        # Проводимость
        sigma = np.zeros(maxSize)

        for layer in layers:
            fillMedium(layer, eps, mu, sigma)

        # Коэффициенты для учета потерь
        loss = sigma * dt / (2 * eps * eps0)
        ceze = (1.0 - loss) / (1.0 + loss)
        cezh = W0 / (eps * (1.0 + loss))

        # Источник
        magnitude = 1.0
        signal = sources.HarmonicPlaneWave.make_continuous(magnitude, f_Hz, dt, Sc,
                                                           eps[sourcePos],
                                                           mu[sourcePos])
        source = sources.SourceTFSF(signal, 0.0, Sc, eps[sourcePos], mu[sourcePos])
        # source = sources.Harmonic.make_continuous(magnitude, f_Hz, dt, Sc)

        Ez = np.zeros(maxSize)
        Hy = np.zeros(maxSize - 1)

        # Создание экземпляров классов граничных условий
        boundary_left = boundary.ABCSecondLeft(eps[0], mu[0], Sc)
        boundary_right = boundary.ABCSecondRight(eps[-1], mu[-1], Sc)

        # Параметры отображения поля E
        display_field = Ez
        display_ylabel = 'Ez, В/м'
        display_ymin = -2.1
        display_ymax = 2.1

        # Создание экземпляра класса для отображения
        # распределения поля в пространстве
        display = tools.AnimateFieldDisplay(dx, dt,
                                            maxSize,
                                            display_ymin, display_ymax,
                                            display_ylabel,
                                            title='fdtd_dielectric')

        display.activate()
        display.drawSources([sourcePos])
        display.drawProbes(probesPos)
        for layer in layers:
            display.drawBoundary(layer.xmin)
            if layer.xmax is not None:
                display.drawBoundary(layer.xmax)

        for t in range(1, maxTime):
            # Расчет компоненты поля H
            Hy = Hy + (Ez[1:] - Ez[:-1]) * Sc / (W0 * mu)

            # Источник возбуждения с использованием метода
            # Total Field / Scattered Field
            Hy[sourcePos - 1] += source.getH(t)

            # Расчет компоненты поля E
            Ez[1:-1] = ceze[1: -1] * Ez[1: -1] + cezh[1: -1] * (Hy[1:] - Hy[: -1])

            # Источник возбуждения с использованием метода
            # Total Field / Scattered Field
            Ez[sourcePos] += source.getE(t)

            boundary_left.updateField(Ez, Hy)
            boundary_right.updateField(Ez, Hy)

            # Регистрация поля в датчиках
            for probe in probes:
                probe.addData(Ez, Hy)

            if t % speed_refresh == 0:
                display.updateData(display_field, t)

        display.stop()

        # Отображение сигнала, сохраненного в пробнике
        res = tools.showProbeSignals(probes, dx, dt, -2.1, 2.1)

        print(res)
        # Расчет необходимых значений
        pattern = r"Max = ([\d.-]+); Min = ([\d.-]+)"
        for i, input_str in enumerate(res):
            match = re.search(pattern, input_str)
            max_value = float(match.group(1))
            min_value = abs(float(match.group(2)))
            if i == 0:
                E_m_ref = max(max_value, min_value)
            elif i == 1:
                E_m_fall = max(max_value, min_value)
            else:
                E_m_pr = max(max_value, min_value)
        G_fdtd = -E_m_ref / E_m_fall
        G_fdtd_list.append(G_fdtd)
        T_fdtd = E_m_pr / E_m_fall
        T_fdtd_list.append(T_fdtd)
        d_g = (G_fdtd - calc.G) / calc.G * 100
        d_g_list.append(d_g)
        d_t = (T_fdtd - calc.T) / calc.T * 100
        d_t_list.append(d_t)
        # Сложные формулы ((
        def b_d_2(e_i):
            return arcsin((sqrt(e_i) / Sc) * sin(calc.pi * Sc / cur_N))
        G_volna = (sqrt(1) * cos(b_d_2(calc.e_2)) - sqrt(calc.e_2) * cos(b_d_2(1))) / (sqrt(1) * cos(b_d_2(calc.e_2)) + sqrt(calc.e_2) * cos(b_d_2(1)))
        G_volna_list.append(G_volna)
        T_volna = (2 * sqrt(1) * cos(b_d_2(1))) / (sqrt(1) * cos(b_d_2(calc.e_2)) + sqrt(calc.e_2) * cos(b_d_2(1)))
        T_volna_list.append(T_volna)
        # Итоговая строчка для экселя
        answer.append(f'{cur_D:.7f}\t{E_m_fall:.5f}\t{E_m_ref:.5f}\t{G_fdtd:.5f}\t{G_volna:.5f}\t{d_g:.5f}\t{E_m_pr:.5f}\t{T_fdtd:.5f}\t{T_volna:.5f}\t{d_t:.5f}')

    # Выводим таблицу для экселя
    for answer_str in answer:
        print(answer_str)

    # Строим графики
    plt.close('all')

    # G - графики
    plt.figure('G')
    plt.plot(calc.N_range, [calc.G] * len(calc.N_range), label='Г', linestyle='-')
    plt.plot(calc.N_range, G_fdtd_list, label='Г_fdtd', linestyle='--')
    plt.plot(calc.N_range, G_volna_list, label='~Г', linestyle='-.')
    plt.legend()
    plt.xlabel('Nl')
    plt.ylabel('Г [б\р]')
    plt.grid(True)
    # d_G
    plt.figure('d_Г')
    plt.plot(calc.N_range, d_g_list, label='d_Г')
    plt.legend()
    plt.xlabel('Nl')
    plt.ylabel('d_Г [%]')
    plt.grid(True)
    # T
    plt.figure('T')
    plt.plot(calc.N_range, [calc.T] * len(calc.N_range), label='T', linestyle='-')
    plt.plot(calc.N_range, T_fdtd_list, label='T_fdtd', linestyle='--')
    plt.plot(calc.N_range, T_volna_list, label='~T', linestyle='-.')
    plt.legend()
    plt.xlabel('Nl')
    plt.ylabel('T [б\р]')
    plt.grid(True)
    # d_T
    plt.figure('d_T')
    plt.plot(calc.N_range, d_t_list, label='d_T')
    plt.legend()
    plt.xlabel('Nl')
    plt.ylabel('d_T [%]')
    plt.grid(True)
    plt.show()