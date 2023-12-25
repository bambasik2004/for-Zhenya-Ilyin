import math
import numpy as np
from Animation import Source1D


class Harmonic(Source1D):
    '''
    Источник, создающий гармонический сигнал
    '''

    def __init__(self, magnitude, Nl, phi_0=None, Sc=1.0):
        '''
        magnitude - максимальное значение в источнике;
        Nl - количество отсчетов на длину волны;
        Sc - число Куранта.
        '''
        self.magnitude = magnitude
        self.Nl = Nl
        self.Sc = Sc

        if phi_0 is None:
            self.phi_0 = -2 * np.pi / Nl
        else:
            self.phi_0 = phi_0

    def getE(self, time):
        return self.magnitude * np.sin(2 * np.pi * self.Sc * time / self.Nl + self.phi_0)

    @staticmethod
    def make_continuous(magnitude: float,
                        freq: float,
                        dt: float,
                        Sc: float) -> 'Harmonic':
        # Количество ячеек на длину волны
        Nl = Sc / (freq * dt)
        phi_0 = -2 * np.pi / Nl
        return Harmonic(magnitude, Nl, phi_0, Sc)


class LayerContinuous:
    def __init__(self,
                 xmin: float,
                 xmax: float = None,
                 eps: float = 1.0,
                 mu: float = 1.0,
                 sigma: float = 0.0):
        self.xmin = xmin
        self.xmax = xmax
        self.eps = eps
        self.mu = mu
        self.sigma = sigma

class LayerDiscrete:
    def __init__(self,
                 xmin: int,
                 xmax: int = None,
                 eps: float = 1.0,
                 mu: float = 1.0,
                 sigma: float = 0.0):
        self.xmin = xmin
        self.xmax = xmax
        self.eps = eps
        self.mu = mu
        self.sigma = sigma

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

