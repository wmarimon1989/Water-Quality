__author__ = 'Efraín Domínguez Calle, PhD - Wilfredo Marimón Bolivar, PhD - Nathalie Toussaint, Ing'
__license__ = "Uso Libre"
__version__ = "1.0"
__maintainer__ = "Wilfredo Marimón Bolivar"
__email__ = 'w.marimon@javeriana.edu.co, edoc@marthmodelling.org'
__status__ = "En desarrollo"

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xlrd
import sys
from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.sample import fast_sampler
from SALib.analyze import fast

def read_sheet(workbook, name):
    # Reading Water quality objetives
    sheet = workbook.sheet_by_name(name)
    # Number of written Rows in sheet
    r = sheet.nrows
    # Number of written Columns in sheet
    c = sheet.ncols
    answ = np.zeros([r - 3, c - 6])
    # Reading each cell in excel sheet
    for i in range(3, r):
        for j in range(6, c):
            answ[i - 3, j - 6] = float(sheet.cell_value(i, j))

    return answ

def read_config_file(config_file, sheet_name_OBCAL='OBCAL', sheet_name_DATA='DATA', sheet_name_CONST='CONST'):
    """
    Reads an Excel configuration file with water quality objetives and data
    :param config_file: The path to cel file
    :param sheet_name_OBCAL: The name of Excel sheet containing water quality objetives set by CAR
    :param sheet_name_Data: The name of Excel sheet containing water quality data
    :param sheet_name_Data: The name of Excel sheet containing reacton constants and others parameters
    :return:
    """
    # Following if is to set a mutable parameter as default parameter
    wb = xlrd.open_workbook(config_file)

    # Reading water quality objetives
    OBCAL = read_sheet(wb, sheet_name_OBCAL)

    # Reading water quality data
    WQD = read_sheet(wb, sheet_name_DATA)

    # Reading water constants
    CONST = read_sheet(wb, sheet_name_CONST)

    return OBCAL, WQD, CONST

if __name__ == '__main__':
    archivo_entrada = 'Prueba_Rio_Sumapaz.xlsx'
    directorio_salida = './salida/'
    show = False
    export = True

def cal_IGAP (SAV):
  C = read_config_file(archivo_entrada)
  OBCAL = C[0]
  WQD = C[1]
  CONST = C[2]

  # Calculating constants rates and other  parameters
  leng = int(len(WQD) / 2)
  Q = np.zeros(leng)
  H = np.zeros(leng)
  B = np.zeros(leng)
  L = np.zeros(leng)

  for i in range(0, (leng)):
    Cau = (WQD[2 * i, 1] + WQD[2 * i + 1, 1]) / 2
    prof = (WQD[2 * i, 2] + WQD[2 * i + 1, 2]) / 2
    Anch = (WQD[2 * i, 3] + WQD[2 * i + 1, 3]) / 2
    Long = (WQD[2 * i, 0] + WQD[2 * i + 1, 0]) / 2
    Q[i] = Cau
    H[i] = prof
    B[i] = Anch
    L[i] = Long

  vel = Q / (B * H) * 84600
  Ksdbo = (SAV[1] * 84.6) / H

  # Calculating the global reaction constants of each water quality parameter
  Rod = (SAV[0]  - CONST[:, 0] - (CONST[:, 1] + CONST[:, 2] + CONST[:, 3]))
  Rdbo = (CONST[:, 0] + Ksdbo)
  Rdqo = CONST[:, 1]
  Rnh3 = (CONST[:, 4] - CONST[:, 5] - CONST[:, 2])
  Rno2 = (CONST[:, 2] - CONST[:, 3])
  Rno3 = (CONST[:, 3] - (CONST[:, 6]))
  Rporg = (CONST[:, 7] + CONST[:, 8])
  RpH = CONST[:, 9]
  Rcon = CONST[:, 10]
  Rsst = Rcon * 0.85

  # Establish the importance weights of each water quality parameter
  Xod = 0.2
  Xph = 0.05
  Xnh3 = 0.1
  Xdbo = 0.1
  Xdqo = 0.1
  Xno2 = 0.1
  Xno3 = 0.1
  Xporg = 0.1
  Xcon = 0.05
  Xsst = 0.1

  # Verification of the correct adjustment of the important weights
  Verf = Xod + Xph + Xnh3 + Xdbo + Xdqo + Xno2 + Xno3 + Xporg + Xcon + Xsst
  Verf = round(Verf, 3)
  if Verf != 1.0:
    sys.exit("Error message.The sum of the weights is not equal to 1.")

  # Estimate individual self-purification index

  IGAP_OD = np.zeros(leng)
  IGAP_DBO = np.zeros(leng)
  IGAP_DQO = np.zeros(leng)
  IGAP_NH3 = np.zeros(leng)
  IGAP_NO2 = np.zeros(leng)
  IGAP_NO3 = np.zeros(leng)
  IGAP_Por = np.zeros(leng)
  IGAP_pH = np.zeros(leng)
  IGAP_Con = np.zeros(leng)
  IGAP_SST = np.zeros(leng)

  for j in range(0, leng):
    IGAP_OD_out = Xod * (((- WQD[2 * j, 6] + WQD[2 * j + 1, 6]) / (OBCAL[j, 2]))) * (L[j] * Rod / vel[j])
    IGAP_DBO_out = Xdbo * ((WQD[2 * j, 7] - WQD[2 * j + 1, 7]) / (OBCAL[j, 3])) * (L[j] * Rdbo / vel[j])
    IGAP_DQO_out = Xdqo * ((WQD[2 * j, 8] - WQD[2 * j + 1, 8]) / (OBCAL[j, 4])) * (L[j] * Rdqo / vel[j])
    IGAP_NH3_out = Xnh3 * ((WQD[2 * j, 11] - WQD[2 * j + 1, 11]) / (OBCAL[j, 7])) * (L[j] * Rnh3 / vel[j])
    IGAP_NO2_out = Xno2 * ((WQD[2 * j, 10] - WQD[2 * j + 1, 10]) / (OBCAL[j, 6])) * (L[j] * Rno2 / vel[j])
    IGAP_NO3_out = Xno3 * ((WQD[2 * j, 9] - WQD[2 * j + 1, 9]) / (OBCAL[j, 5])) * (L[j] * Rno3 / vel[j])
    IGAP_Por_out = Xporg * ((WQD[2 * j, 13] - WQD[2 * j + 1, 13]) / (OBCAL[j, 9])) * (L[j] * Rporg / vel[j])
    IGAP_pH_out = Xph * ((WQD[2 * j, 4] - WQD[2 * j + 1, 4]) / (OBCAL[j, 0])) * (L[j] * RpH / vel[j])
    IGAP_Con_out = Xcon * ((WQD[2 * j, 5] - WQD[2 * j + 1, 5]) / (OBCAL[j, 1])) * (L[j] * Rcon / vel[j])
    IGAP_SST_out = Xsst * ((WQD[2 * j, 12] - WQD[2 * j + 1, 12]) / (OBCAL[j, 8])) * (L[j] * Rsst / vel[j])

    IGAP_OD[j] = IGAP_OD_out[j]
    IGAP_DBO[j] = IGAP_DBO_out[j]
    IGAP_DQO[j] = IGAP_DQO_out[j]
    IGAP_NH3[j] = IGAP_NH3_out[j]
    IGAP_NO2[j] = IGAP_NO2_out[j]
    IGAP_NO3[j] = IGAP_NO3_out[j]
    IGAP_Por[j] = IGAP_Por_out[j]
    IGAP_pH[j] = IGAP_pH_out[j]
    IGAP_Con[j] = IGAP_Con_out[j]
    IGAP_SST[j] = IGAP_SST_out[j]

  # Estimate Global self-purification index
  IGAP = IGAP_OD + IGAP_DBO + IGAP_NH3 + IGAP_NO2 + IGAP_NO3 + IGAP_Por + IGAP_pH + IGAP_Con + IGAP_SST
  IGAP = np.where(IGAP > 3, 3, IGAP)
  IGAP = np.where(IGAP < -3, -3, IGAP)
  return IGAP

#Perform sensitivity analysis
problem = {'num_vars': 2,
           'names': ['Ka', 'Vs'],
           'bounds': [[0.00001, 1],
                     [0.1, 10]]
           }

# Generate samples
param_values = fast_sampler.sample(problem, 1000)
DATA = read_config_file(archivo_entrada)
leng2 = int(len(DATA[1]) / 2)
leng1 = param_values.shape[0]

# Run model (example)
Y = np.zeros((leng1, leng2))
for i, SAV in enumerate(param_values):
    Y[i] = cal_IGAP(SAV)

# Perform the analysis
S1 = np.zeros((leng2, 2))
#S1_conf = np.zeros((leng2, 2))
ST = np.zeros((leng2, 2))
#ST_conf = np.zeros((leng2, 2))

for j in range(leng2):
      Si = fast.analyze(problem, Y[:,j], print_to_console=False)
      S1[j,:] = Si['S1']
      #S1_conf[j, :] = Si['S1_conf']
      ST[j, :] = Si['ST']
      #ST_conf[j,:] = Si['ST_conf']

IGAP = cal_IGAP(SAV)
SA = np.concatenate((S1, ST), axis=1)
Salida_IGAP = pd.DataFrame(IGAP, columns=['IGAP'])
Salida_IGAP.to_excel(r'C:\Users\willy\OneDrive\Escritorio\Curso_Python\IGAP\IGAP.xlsx', index=False, header=True)
Salida_SA = pd.DataFrame(SA, columns=['S1_VAR1', 'S1_VAR2', 'ST_VAR1', 'ST_VAR2'])
Salida_SA.to_excel(r'C:\Users\willy\OneDrive\Escritorio\Curso_Python\IGAP\Sensitivity_Analysis.xlsx', index=False, header=True)
plt.bar(range(1, 36, 1), IGAP, edgecolor='black')
plt.title("Calculation IGAP")
plt.xlabel("River section")
plt.ylabel("IGAP")
plt.show()