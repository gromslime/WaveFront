import os
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.signal import detrend
from matplotlib import cm

class CONSTANTS:
    fQ = 500
    Radius = 20 # mm
    ApertureX = 0 # mm
    ApertureY = 0 # mm
    ApertureRadius = 20 # mm
    C11 = (0.022e-6)*5**0.5
def mainAction(apertureShiftX):
    CONSTANTS.ApertureX = apertureShiftX
    # X, Y series
    X = np.linspace(-CONSTANTS.Radius, CONSTANTS.Radius, CONSTANTS.fQ)
    Y = np.linspace(-CONSTANTS.Radius, CONSTANTS.Radius, CONSTANTS.fQ)

    # Mesh realization
    X,Y = np.meshgrid(X,Y)
    R = (X**2 + Y**2) ** 0.5
    Z = CONSTANTS.C11 * (6*R**4 + 6*R**2 + 1)

    # Aperture realization
    radiusApertureVector = (CONSTANTS.ApertureX**2 + CONSTANTS.ApertureY**2)**0.5
    _indexes = np.where(((X + CONSTANTS.ApertureX)**2 + (Y + CONSTANTS.ApertureY)**2)**0.5 > CONSTANTS.ApertureRadius)
    Z[_indexes] = np.nan


    dZy, dZx = np.gradient(Z)
    dY, dX = np.gradient(X)


    Der=dZx/dX

    mean_index = 250
    Der_mean = Der[250]

    Deviation = 0
    k = 0
    for i, val in enumerate(Der):
        if math.isnan(Der_mean[i]):
            continue
        Deviation += np.arctan(Der_mean[i])
        k+= 1
    Deviation = Deviation/k
    ##print(k)

    Fi=Deviation*3600
    # print('Отклонение поверхности измеренное АК =',Fi,'угл.сек.')

    # fig, ax = plt.subplots(subplot_kw = {"projection": "3d"})
    # ax.plot_surface(X,Y,Z, cmap = cm.Blues)
    return Fi


for distance in [200,500,1000,1200,2000]:
    X = []
    Y = []
    for i in np.linspace(-1300, 1300, 40):
        dX = distance * np.tan(math.radians(i/3600))
        Y.append(mainAction(dX))
        X.append(i)
    Y = detrend(Y)
    plt.plot(X,Y, label = distance)
plt.legend()
plt.show()