__author__ = 'Wilfredo Marimón Bolivar, PhD'
__license__ = "Uso Libre"
__version__ = "1.0"
__email__ = 'w.marimon@javeriana.edu.co'
__status__ = "En desarrollo"

import numpy as np
import matplotlib.pyplot as plt

#Definicion del segmento y la geometria del reactor
#Vol = float(input("Volumen en m3: "))
Vol = 5
#Q = float(input("Caudal en m3/s: "))
Q = 0.01
#prof = float(input("Profundidad del agua en metros: "))
prof = 3
#v = float(input("Valocidad del agua en m/s: "))
v = np.arange(1, 11, 1)
A = Vol/prof

#Definicion del tiempo de analisis
tini = 0
tfinal = 3600
t = np.arange(tini, tfinal, 1)

#Condiciones iniciales
ci_ibu = 0.1

#Constantes de reaccion y parametros del medio
Us = 0.01
Kaw = 0.001                           #Total air–water mass transfer rate constant
Kddpc = 0.001                         #Dry deposition of particulate-bound compounds
Ksr = 0.0001                          #Sediment resuspension
Kphto = 0.0001                        #Phtodegradation rate constant
Cgas = 1e-5                           #Constante de solubilidad en gas
Henry = 0.45                          #Constante de Henry para el sistema estudiado
R = 8.314                             #Constante universal de los gases
T = 298                               #Temperatura
Prec = 0.1                            #Precipitacion media
Cprec = 1e-9
Ku = 1e-5
Bp = 0.5
Kd = 1e-3
Kg = 1e-4
Cp = 1e-10
Cs = 1e-11
Ksed = 0.005
Kss = 1e-3

Kdawe = Kaw * Cgas / (Henry/R * T)  #Diffusive air–water exchange
Kwd = Prec * Cprec                  #Wet deposition
Kwpe = Ku * Bp * prof               #Water–phytoplankton exchange
Krph = - (Kd + Kg) * prof * Cp      #uptake phytoplacton
Kssin = Cs/Ksed                     #Sediment sinking

ktotal = (Kaw + Kss - Kwpe + Ksr + Kphto)

#Calculo de la concentracion de salida

Cf = np.zeros((len(t), len(v)))

for i in range(len(v)):
    a = -(ktotal + Q + (A * Us)) / Vol
    b = ((Q*ci_ibu) + Kwd + Kwpe + Krph + Kssin) / Vol
    cout_ibu = (((np.exp(a*t)) * ((a * ci_ibu) + b)) - b)/a
    cout_ibu = np.where(cout_ibu < 0, 0, cout_ibu)
    Cf[:,i] = cout_ibu

#Generar graficas
plt.plot(t, Cf[:,0], 'b', t, Cf[:,3], 'r', t, Cf[:,4], 'g')
plt.xlabel("Tiempo (seg)")
plt.ylabel("Concentración Ibuprofeno (mg/L)")
#plt.legend(('Ktotal = 0.1', 'Ktotal = 0.4', 'Ktotal = 0.9'),
#prop = {'size': 10}, loc='upper right')
plt.show()
