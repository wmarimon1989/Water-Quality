__author__ = 'Wilfredo Marimón Bolivar, PhD'
__license__ = "Uso Libre"
__version__ = "1.0"
__email__ = 'w.marimon@javeriana.edu.co'
__status__ = "En desarrollo"

import numpy as np
import matplotlib.pyplot as plt

#Definicion del segmento y la geometria del reactor
Vol = float(input("Volumen en m3: "))
Q = float(input("Caudal en m3/s: "))
prof = float(input("Profundidad del agua en metros: "))
A = Vol/prof

#Definicion del tiempo de analisis
tini = 0
tfinal = 60
t = np.arange(tini, tfinal, 1)

#Condiciones iniciales
ci_ibu = 0.1

#Constantes de reaccion y parametros del medio
Us = 0.01
ktotal = np.arange(0.1, 1.1, 0.1)

#Calculo de la concentracion de salida
Cf = np.zeros((len(t), len(ktotal)))

for i in range(len(ktotal)):
    a = -(ktotal[i] + ((Q + A * Us) / Vol))
    b = Q * ci_ibu / Vol
    cout_ibu = (((np.exp(a*t)) * ((a * ci_ibu) + b)) - b)/a
    cout_ibu = np.where(cout_ibu < 0, 0, cout_ibu)
    Cf[:,i] = cout_ibu

#Generar graficas
plt.plot(t, Cf[:,0], 'b', t, Cf[:,3], 'r', t, Cf[:,8], 'g')
plt.xlabel("Tiempo (seg)")
plt.ylabel("Concentración Ibuprofeno (mg/L)")
plt.legend(('Ktotal = 0.1', 'Ktotal = 0.4', 'Ktotal = 0.9'),
prop = {'size': 10}, loc='upper right')
plt.show()
