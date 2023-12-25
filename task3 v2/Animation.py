import pylab
import numpy as np
import numpy.typing as npt
from abc import ABCMeta, abstractmethod
from typing import List, Optional

class Probe:
    '''
    Класс для хранения временного сигнала в пробнике.
    '''

    def __init__(self, position: int, maxTime: int):
        '''
        position - положение пробника (номер ячейки).
        maxTime - максимально количество временных шагов для хранения в пробнике.
        '''
        self.position = position

        # Временные сигналы для полей E и H
        self.E = np.zeros(maxTime)
        self.H = np.zeros(maxTime)

        # Номер временного шага для сохранения полей
        self._time = 0

    def addData(self, E: npt.NDArray[float], H: npt.NDArray[float]):
        '''
        Добавить данные по полям E и H в пробник.
        '''
        self.E[self._time] = E[self.position]
        self.H[self._time] = H[self.position]
        self._time += 1

class AnimateFieldDisplay:
    '''
    Класс для отображения анимации распространения ЭМ волны в пространстве
    '''

    def __init__(self,
                 dx: float,
                 dt: float,
                 maxXSize: int,
                 minYSize: float, maxYSize: float,
                 yLabel: str,
                 title: Optional[str] = None
                 ):
        '''
        dx - дискрет по простарнству, м
        dt - дискрет по времени, сек
        maxXSize - размер области моделирования в отсчетах.
        minYSize, maxYSize - интервал отображения графика по оси Y.
        yLabel - метка для оси Y
        '''
        self.maxXSize = maxXSize
        self.minYSize = minYSize
        self.maxYSize = maxYSize
        self._xList = None
        self._line = None
        self._xlabel = 'x, м'
        self._ylabel = yLabel
        self._probeStyle = 'xr'
        self._sourceStyle = 'ok'
        self._dx = dx
        self._dt = dt
        self._title = title

    def activate(self):
        '''
        Инициализировать окно с анимацией
        '''
        self._xList = np.arange(self.maxXSize) * self._dx

        # Включить интерактивный режим для анимации
        pylab.ion()

        # Создание окна для графика
        self._fig, self._ax = pylab.subplots(
            figsize=(10, 6.5))

        if self._title is not None:
            self._fig.suptitle(self._title)

        # Установка отображаемых интервалов по осям
        self._ax.set_xlim(0, self.maxXSize * self._dx)
        self._ax.set_ylim(self.minYSize, self.maxYSize)

        # Установка меток по осям
        self._ax.set_xlabel(self._xlabel)
        self._ax.set_ylabel(self._ylabel)

        # Включить сетку на графике
        self._ax.grid()

        # Отобразить поле в начальный момент времени
        self._line, = self._ax.plot(self._xList, np.zeros(self.maxXSize))

    def drawProbes(self, probesPos: List[int]):
        '''
        Нарисовать пробники.

        probesPos - список координат пробников для регистрации временных
            сигналов (в отсчетах).
        '''
        # Отобразить положение пробника
        self._ax.plot(np.array(probesPos) * self._dx,
                      [0] * len(probesPos), self._probeStyle)

        for n, pos in enumerate(probesPos):
            self._ax.text(
                pos * self._dx,
                0,
                '\n{n}'.format(n=n + 1),
                verticalalignment='top',
                horizontalalignment='center')

    def drawSources(self, sourcesPos: List[int]):
        '''
        Нарисовать источники.

        sourcesPos - список координат источников (в отсчетах).
        '''
        # Отобразить положение пробника
        self._ax.plot(np.array(sourcesPos) * self._dx,
                      [0] * len(sourcesPos), self._sourceStyle)

    def drawBoundary(self, position: int):
        '''
        Нарисовать границу в области моделирования.

        position - координата X границы (в отсчетах).
        '''
        self._ax.plot([position * self._dx, position * self._dx],
                      [self.minYSize, self.maxYSize],
                      '--k')

    def stop(self):
        '''
        Остановить анимацию
        '''
        pylab.ioff()

    def updateData(self, data: npt.NDArray[float], timeCount: int):
        '''
        Обновить данные с распределением поля в пространстве
        '''
        self._line.set_ydata(data)
        time_str = '{:.5f}'.format(timeCount * self._dt * 1e9)
        self._ax.set_title(f'{time_str} нс')
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

class Source1D(metaclass=ABCMeta):
    '''
    Базовый класс для всех источников одномерного метода FDTD
    '''

    @abstractmethod
    def getE(self, time):
        '''
        Метод должен возвращать значение поля источника в момент времени time
        '''
        pass

    def getH(self, time):
        return 0.0

class SourcePlaneWave(metaclass=ABCMeta):
    @abstractmethod
    def getE(self, position, time):
        pass

class Source(Source1D):
    def __init__(self, source: SourcePlaneWave,
                 sourcePos: float,
                 Sc: float = 1.0,
                 eps: float = 1.0,
                 mu: float = 1.0):
        self.source = source
        self.sourcePos = sourcePos
        self.Sc = Sc
        self.eps = eps
        self.mu = mu
        self.W0 = 120.0 * np.pi

    def getH(self, time):
        return -(self.Sc / (self.W0 * self.mu)) * (
                    self.source.getE(self.sourcePos, time) - self.source.getE(self.sourcePos - 1, time))

    def getE(self, time):
        return (self.Sc / np.sqrt(self.eps * self.mu)) * (
                    self.source.getE(self.sourcePos - 0.5, time + 0.5) + self.source.getE(self.sourcePos + 0.5,
                                                                                          time + 0.5))