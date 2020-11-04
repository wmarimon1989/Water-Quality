from __future__ import print_function
from __future__ import absolute_import

__author__ = 'Efraín Domínguez Calle, PhD - Wilfredo Marimón Bolivar, PhD'
__copyright__ = "Copyright 2017, Mathmodelling"
__credits__ = ["Efraín Domínguez Calle"]
__license__ = "Uso Libre"
__version__ = "1.0"
__email__ = 'edoc@marthmodelling.org, w.marimon@javeriana.edu.co'
__status__ = "En desarrollo"


from builtins import range
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import xlrd
import xlwt
from util import join
from util import used_vars
BOLD_FONT_XLWT = xlwt.Style.easyxf('font: bold on;')

def read_sheet(workbook, name):
    # Reading water depth sheet
    sheet = workbook.sheet_by_name(name)
    # Number of written Rows in sheet
    r = sheet.nrows
    # Number of written Columns in sheet
    c = sheet.ncols
    answ = np.zeros([r - 1, c])
    # Reading each cell in excel sheet 'BC'
    for i in range(1, r):
        for j in range(c):
            answ[i - 1, j] = float(sheet.cell_value(i, j))

    return answ

def plot(ax, name, data):
    ax.tick_params(labelsize=6)
    ax.yaxis.get_offset_text().set_fontsize(6)
    ax.plot(data[0], data[1])
    ax.set_title(name, fontsize=8)

def save_plot(plt, title, xlabel, ylabel, data, path):
    plt.plot(data[0], data[1])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig("%s.png" % join(path, title), dpi=300)
    plt.clf()

def save_sheet(book, name, data):
    sheet = book.add_sheet(name)

    for i in range(1, len(data) + 1):
        sheet.write(i, 0, i - 1, BOLD_FONT_XLWT)
    for i in range(1, len(data[0]) + 1):
        sheet.write(0, i, i - 1, BOLD_FONT_XLWT)

    for i in range(1, len(data) + 1):
        for j in range(1, len(data[0]) + 1):
            sheet.write(i, j, data[i -1 , j - 1])

def read_config_file(config_file, sheet_name_wd='WD', sheet_name_sl='SL', sheet_name_wv='WV', sheet_name_bc='BC',
                     sheet_name_ic='IC', sheet_name_ST='ST', sheet_name_SOD='SOD', sheet_name_SDBO='SDBO', sheet_name_SNH3='SNH4',
                     sheet_name_SNO2='SNO2', sheet_name_SNO3='SNO3', sheet_name_STDS='STDS', sheet_name_SGyA='SGyA',
                     sheet_name_SDQO='SDQO', sheet_name_SPdis='SPdis', sheet_name_SPorg='SPorg', sheet_name_SEC='SEC',
                     sheet_name_STC='STC', sheet_name_STSS='STSS', sheet_name_SSS='SSS', sheet_name_SpH='SpH',
                     sheet_name_SALK='SALK', sheet_name_Caudales='Caudales'):
    """
    Reads an Excel configuration file with initial and boundary conditions and also time series for sinks and sources
    :param config_file: The path to
    cel file
    :param sheet_name_wd: The name of Excel sheet containing water depth for each cross section in the water channel
    :param sheet_name_sl: The name of Excel sheet containing bed slope for each cross section in the water channel
    :param sheet_name_wv: The name of Excel sheet containing water velocity for each cross section in the water channel
    :param sheet_name_bc: The name of Excel sheet with boundary condition information by default is 'BC'
    :param sheet_name_ic: The name of Excel sheet with initial condition information by default is 'IC'
    :param sheet_names_list_sources: A list with sheet names with the information about sinks and sources, default['S1']
    :return:
    """
    # Following if is to set a mutable parameter as default parameter
    wb = xlrd.open_workbook(config_file)

    # Reading water depth sheet
    wd = read_sheet(wb, sheet_name_wd)

    # Reading water bed slope
    sl = read_sheet(wb, sheet_name_sl)

    # Reading water velocities sheet
    wv = read_sheet(wb, sheet_name_wv)

    # Reading boundary conditions sheet
    bc = read_sheet(wb, sheet_name_bc)

    # Reading initial conditions sheet
    ic = read_sheet(wb, sheet_name_ic)

    # Reading sinks and sources sheet
    ST = read_sheet(wb, sheet_name_ST)

    # Reading sinks and sources sheet
    SOD = read_sheet(wb, sheet_name_SOD)

    # Reading sinks and sources sheet
    SDBO = read_sheet(wb, sheet_name_SDBO)

    # Reading sinks and sources sheet
    SNO2 = read_sheet(wb, sheet_name_SNO2)

    # Reading sinks and sources sheet
    SNO3 = read_sheet(wb, sheet_name_SNO3)

    # Reading sinks and sources sheet
    SNH3 = read_sheet(wb, sheet_name_SNH3)

    # Reading sinks and sources sheet
    STDS = read_sheet(wb, sheet_name_STDS)

    # Reading sinks and sources sheet
    SGyA = read_sheet(wb, sheet_name_SGyA)

    # Reading sinks and sources sheet
    SDQO = read_sheet(wb, sheet_name_SDQO)

    # Reading sinks and sources sheet
    SPorg = read_sheet(wb, sheet_name_SPorg)

    # Reading sinks and sources sheet
    SPdis = read_sheet(wb, sheet_name_SPdis)

    # Reading sinks and sources sheet
    SEC = read_sheet(wb, sheet_name_SEC)

    # Reading sinks and sources sheet
    STC = read_sheet(wb, sheet_name_STC)

    # Reading sinks and sources sheet
    STSS = read_sheet(wb, sheet_name_STSS)

    # Reading sinks and sources sheet
    SSS = read_sheet(wb, sheet_name_SSS)

    # Reading sinks and sources sheet
    SpH = read_sheet(wb, sheet_name_SpH)

    # Reading sinks and sources sheet
    SALK = read_sheet(wb, sheet_name_SALK)


    # Leer caudales primera fila no puede contener ceros
    # La verificación se hace cuando se carga el archivo
    Caudales = read_sheet(wb, sheet_name_Caudales)

    SOD = SOD[0:, 1:] * Caudales[0:, 1:] / (Caudales[0, 1:] + Caudales[0:, 1:])
    SDBO = SDBO[0:, 1:] * Caudales[0:, 1:] / (Caudales[0, 1:] + Caudales[0:, 1:])
    SNH3 = SNH3[0:, 1:] * Caudales[0:, 1:] / (Caudales[0, 1:] + Caudales[0:, 1:])
    SNO2 = SNO2[0:, 1:] * Caudales[0:, 1:] / (Caudales[0, 1:] + Caudales[0:, 1:])
    SNO3 = SNO3[0:, 1:] * Caudales[0:, 1:] / (Caudales[0, 1:] + Caudales[0:, 1:])
    SDQO = SDQO[0:, 1:] * Caudales[0:, 1:] / (Caudales[0, 1:] + Caudales[0:, 1:])
    STDS = STDS[0:, 1:] * Caudales[0:, 1:] / (Caudales[0, 1:] + Caudales[0:, 1:])
    SGyA = SGyA[0:, 1:] * Caudales[0:, 1:] / (Caudales[0, 1:] + Caudales[0:, 1:])
    SEC = SEC[0:, 1:] * Caudales[0:, 1:] / (Caudales[0, 1:] + Caudales[0:, 1:])
    STC = STC[0:, 1:] * Caudales[0:, 1:] / (Caudales[0, 1:] + Caudales[0:, 1:])
    SPorg = SPorg[0:, 1:] * Caudales[0:, 1:] / (Caudales[0, 1:] + Caudales[0:, 1:])
    SPdis = SPdis[0:, 1:] * Caudales[0:, 1:] / (Caudales[0, 1:] + Caudales[0:, 1:])
    STSS = STSS[0:, 1:] * Caudales[0:, 1:] / (Caudales[0, 1:] + Caudales[0:, 1:])
    SSS = SSS[0:, 1:] * Caudales[0:, 1:] / (Caudales[0, 1:] + Caudales[0:, 1:])
    ST = (ST[0:, 1:] + 273)
    SpH = (10 ** (-1 * (SpH[0:, 1:]))) * Caudales[0:, 1:] / (Caudales[0, 1:] + Caudales[0:, 1:])
    SALK = SALK[0:, 1:] * Caudales[0:, 1:] / (Caudales[0, 1:] + Caudales[0:, 1:])

    return wd, sl, wv, bc, ic, ST, SOD, SDBO, SNH3, SNO2, SNO3, STDS, SGyA, SDQO, SPdis, SPorg, SEC, STC, STSS, SSS, SpH, SALK, Caudales


def calidad_explicito(D, dx, ci_T, ci_OD, ci_DBO, ci_NH3, ci_NO2, ci_NO3, ci_DQO, ci_TDS, ci_EC, ci_TC, ci_GyA, ci_Porg, ci_Pdis, ci_TSS,
                      ci_SS, ci_pH, ci_ALK, v, d, S_T, S_OD, S_DBO, S_NH3, S_NO2, S_NO3, S_DQO, S_TDS, S_EC, S_TC, S_GyA, S_Porg, S_Pdis, S_TSS, S_SS,
                      S_pH, S_ALK, Caudales, variables):
    """
    Esta función modela la transición de la concentración del momento t al momento t + dt para todos los
    nodos espaciales de la corriente superficial

    :param ci: matrix (bidimensional) de concentración inicial en el canal y su respectiva distancia x la concentración
    va en g/m3, la distancia en metros
    :param v: vector de velocidad promedio del agua en m/s, tiene las velocidades promedio para cada sección
     y para cada momento de tiempo
    :param d: vector de coeficiente de difusión
    :return: c, dt: la concentración del contaminante en todos los nodos x para el momento de tiempo t + dt y el valor
    de dt que cumple la condición de estabilidad de Courant o CFL

    """
    c_T = ci_T
    c_OD = ci_OD
    c_DBO = ci_DBO
    c_NH3 = ci_NH3
    c_NO2 = ci_NO2
    c_NO3 = ci_NO3
    c_DQO = ci_DQO
    c_TDS = ci_TDS
    c_EC = ci_EC
    c_TC = ci_TC
    c_GyA = ci_GyA
    c_Porg = ci_Porg
    c_Pdis = ci_Pdis
    c_TSS = ci_TSS
    c_SS = ci_SS
    c_pH = ci_pH
    c_ALK = ci_ALK

    maxv = abs(np.max(v))
    maxd = abs(np.max(d))
    pe = maxv * dx / maxd
    if (np.abs(pe) >= 3) or (maxd == 0):
        dt = dx / maxv
        d = d * 0
        #print "Se desconectó la Difusión, el número de peclet es: %s. Courant es igual a: %s. El paso en el " \
              #"tiempo es de: %s segundos" % (round(pe, 2), str(maxv * (dt / dx)), dt)
    elif (np.min(np.abs(pe)) >= 0.1) or (maxd == 0):
        dt = 1 / (2 * maxd / (dx ** 2) + (maxv / dx))
        #print "Se se tienen en cuenta difusion y adveccion, el número de peclet es: %s. Courant es igual a: %s. El paso en el tiempo es de: %s segundos" % (
        #round(pe, 2), str(maxv * (dt / dx)), dt)
    else:
        dt = (dx * dx) / (2 * maxd)
        #print 'Se calcula advección y difusión, el número de peclet es: %s. Courant es igual a: %s. El paso en el ' \
              #'tiempo es de: %s segundos' % (round(pe, 2), str(maxv * (dt / dx)), dt)

    # tfactor es un factor multiplicador del numero de nodos en el tiempo para llegar de t a t + dt, tfactor >= 1,
    # se recomienda aumentarlo de 10 en 10 {10, 100, 1000, 10000... }
    # ## AGREGAR TFACTOR
    tfactor = variables['tfactor']
    # Se guarda el dt inicial como dtini
    dtini = dt
    # Se ajusta el dt según tfactor, dt se hace más pequeño tfactor-veces
    dt = dt / tfactor
    #CONSTANTES
    ki = np.where(v > 0, 0, 1)
    kr = np.where(v < 0, 0, 1)

    #VARIABLES ADICIONALES
    den = variables['den']
    Cp = variables['Cp']
    As1 = variables['As1']
    Jsn = variables['Jsn']
    sbc = variables['sbc']
    Tair = variables['Tair']
    Aair = variables['Aair']
    eair = variables['eair']
    RL = variables['RL']
    Uw = variables['Uw']
    es = variables['es']
    Kw = variables['Kw']
    K1 = variables['K1']
    K2 = variables['K2']
    Vv = variables['Vv']
    As = variables['As']
    CO2S = variables['CO2S']
    Wrp = variables['Wrp']
    FrH = variables['FrH']
    Da = variables['Da']
    ko2 = variables['ko2']
    cs = variables['cs']
    knh3 = variables['knh3']
    ksnh3 = variables['ksnh3']
    alfa_nh3 = variables['alfa_nh3']
    kdbo = variables['kdbo']
    ks = variables['ks']
    alfa_no2 = variables['alfa_no2']
    ksod = variables['ksod']
    knt = variables['knt']
    NT = variables['NT']
    kno2 = variables['kno2']
    kno3 = variables['kno3']
    kDQO = variables['kDQO']
    kTDS = variables['kTDS']
    A = variables['A']
    alfa_1 = variables['alfa_1']
    miu = variables['miu']
    F = variables['F']
    kTC = variables['kTC']
    teta_TC = variables['teta_TC']
    kEC = variables['kEC']
    teta_EC = variables['teta_EC']
    Jdbw = variables['Jdbw']
    qtex = variables['qtex']
    kN = variables['kN']
    kH = variables['kH']
    kOH = variables['kOH']
    fdw = variables['fdw']
    kf = variables['kf']
    kb = variables['kb']
    kv = variables['kv']
    Cg = variables['Cg']
    Henry = variables['Henry']
    R = variables['R']
    T = variables['T']
    alfa_2 = variables['alfa_2']
    resp = variables['resp']
    kPorg = variables['kPorg']
    kPsed = variables['kPsed']
    sigma2 = variables['sigma2']
    Ws = variables['Ws']
    Rs = variables['Rs']
    Rp = variables['Rp']
    teta_DBO = variables['teta_DBO']
    teta_NH3 = variables['teta_NH3']
    teta_NO2 = variables['teta_NO2']
    teta_DQO = variables['teta_DQO']
    teta_NT = variables['teta_NT']
    teta_NO3 = variables['teta_NO3']

    cout_T = c_T
    cout_OD = c_OD
    cout_DBO = c_DBO
    cout_NH3 = c_NH3
    cout_NO2 = c_NO2
    cout_NO3 = c_NO3
    cout_DQO = c_DQO
    cout_TDS = c_TDS
    cout_EC = c_EC
    cout_TC = c_TC
    cout_GyA = c_GyA
    cout_Porg = c_Porg
    cout_Pdis = c_Pdis
    cout_TSS = c_TSS
    cout_SS = c_SS
    cout_pH = c_pH
    cout_ALK = c_ALK

    # range(int(dtini / dt)) determina el número de nodos temporales necesarios para llegar t + dt de forma estable
    for i in range(int(dtini / dt)):
        caudales = Caudales[0:, 1:] / (Caudales[0, 1:] + Caudales[0:, 1:])
        adv_T = -((ki[2:] * v[2:] * c_T[2:] - ki[1:-1] * v[1:-1] * c_T[1:-1]) * (dt / dx) +
                   (kr[1:-1] * v[1:-1] * c_T[1:-1] - kr[0:-2] * v[0:-2] * c_T[0:-2]) * (dt / dx))
        reac_T = (Jsn + sbc*((Tair + 273)**4)*(Aair + 0.031*((eair)**0.5))*(1-RL) - 0.97*sbc*((c_T[0:-2])**4) -
                  0.47*(19 + (0.95*(Uw**2)))*((c_T[0:-2]) - Tair - 273.15) - (19 + (0.95*(Uw**2)))*(es - eair))*D/(den*Cp*As1)
        cout_T[1:-1] = c_T[1:-1] + adv_T + reac_T + ((S_T[1:-1] - c_T[1:-1])*caudales[1:-1, 1])

        adv_OD = -((ki[2:] * v[2:] * c_OD[2:] - ki[1:-1] * v[1:-1] * c_OD[1:-1]) * (dt / dx) +
                   (kr[1:-1] * v[1:-1] * c_OD[1:-1] - kr[0:-2] * v[0:-2] * c_OD[0:-2]) * (dt / dx))
        dif_OD = 0.5 * (d[2:] * c_OD[2:] - 2 * d[1:-1] * c_OD[1:-1] + d[0:-2] * c_OD[0:-2]) * (dt / dx ** 2)
        p = (c_OD[0:-2]) / ((c_OD[0:-2]) + ks)
        reac_OD = (Da + ko2 * (cs - c_OD[0:-2]) - kdbo * c_DBO[0:-2] * p * (teta_DBO ** (c_T[0:-2] - 293.15)) - alfa_nh3 * knh3 *
                   c_NH3[0:-2] * p * (teta_NH3 ** (c_T[0:-2] - 293.15)) - alfa_no2 * kno2 * c_NO2[0:-2] * p * (
                   teta_NO2 ** (c_T[0:-2] - 293.15)) - ksod / D) * dt
        cout_OD[1:-1] = c_OD[1:-1] + adv_OD + dif_OD + reac_OD + ((S_OD[1:-1] - c_OD[1:-1])*caudales[1:-1, 1])
        cout_OD = np.where(cout_OD < 0, 0, cout_OD)

        adv_DBO = -((ki[2:] * v[2:] * c_DBO[2:] - ki[1:-1] * v[1:-1] * c_DBO[1:-1]) * (dt / dx) +
                    (kr[1:-1] * v[1:-1] * c_DBO[1:-1] - kr[0:-2] * v[0:-2] * c_DBO[0:-2]) * (dt / dx))
        dif_DBO = 0.5 * (d[2:] * c_DBO[2:] - 2 * d[1:-1] * c_DBO[1:-1] + d[0:-2] * c_DBO[0:-2]) * (dt / dx ** 2)
        reac_DBO = (-kdbo * c_DBO[0:-2] * p * (teta_DBO ** (c_T[0:-2] - 293.15))) * dt
        cout_DBO[1:-1] = c_DBO[1:-1] + adv_DBO + dif_DBO + reac_DBO + ((S_DBO[1:-1] - c_DBO[1:-1])*caudales[1:-1, 1])

        adv_NH3 = -((ki[2:] * v[2:] * c_NH3[2:] - ki[1:-1] * v[1:-1] * c_NH3[1:-1]) * (dt / dx) +
                    (kr[1:-1] * v[1:-1] * c_NH3[1:-1] - kr[0:-2] * v[0:-2] * c_NH3[0:-2]) * (dt / dx))
        dif_NH3 = 0.5 * (d[2:] * c_NH3[2:] - 2 * d[1:-1] * c_NH3[1:-1] + d[0:-2] * c_NH3[0:-2]) * (dt / dx ** 2)
        reac_NH3 = (knt * NT * (teta_NT ** (c_T[0:-2] - 293.15)) - knh3 * c_NH3[0:-2] * p * (
        teta_NH3 ** (c_T[0:-2] - 293.15)) + ksnh3 / D - F * alfa_1 * miu * A) * dt
        cout_NH3[1:-1] = c_NH3[1:-1] + adv_NH3 + dif_NH3 + reac_NH3 + ((S_NH3[1:-1] - c_NH3[1:-1])*caudales[1:-1, 1])
        cout_NH3 = np.where(cout_NH3 < 0, 0, cout_NH3)

        adv_NO2 = -((ki[2:] * v[2:] * c_NO2[2:] - ki[1:-1] * v[1:-1] * c_NO2[1:-1]) * (dt / dx) +
                    (kr[1:-1] * v[1:-1] * c_NO2[1:-1] - kr[0:-2] * v[0:-2] * c_NO2[0:-2]) * (dt / dx))
        dif_NO2 = 0.5 * (d[2:] * c_NO2[2:] - 2 * d[1:-1] * c_NO2[1:-1] + d[0:-2] * c_NO2[0:-2]) * (dt / dx ** 2)
        reac_NO2 = (knh3 * c_NH3[0:-2] * p * (teta_NH3 ** (c_T[0:-2] - 293.15)) - kno2 * c_NO2[0:-2] * p * (
        teta_NO2 ** (c_T[0:-2] - 293.15)) + kno3 * c_NO3[0:-2] * (teta_NO3 ** (c_T[0:-2] - 293.15))) * dt
        cout_NO2[1:-1] = c_NO2[1:-1] + adv_NO2 + dif_NO2 + reac_NO2 + ((S_NO2[1:-1] - c_NO2[1:-1])*caudales[1:-1, 1])

        adv_NO3 = -((ki[2:] * v[2:] * c_NO3[2:] - ki[1:-1] * v[1:-1] * c_NO3[1:-1]) * (dt / dx) +
                    (kr[1:-1] * v[1:-1] * c_NO3[1:-1] - kr[0:-2] * v[0:-2] * c_NO3[0:-2]) * (dt / dx))
        dif_NO3 = 0.5 * (d[2:] * c_NO3[2:] - 2 * d[1:-1] * c_NO3[1:-1] + d[0:-2] * c_NO3[0:-2]) * (dt / dx ** 2)
        reac_NO3 = (kno2 * c_NO2[0:-2] * p * (teta_NO2 ** (c_T[0:-2] - 293.15)) - kno3 * c_NO3[0:-2] * (
        teta_NO3 ** (c_T[0:-2] - 293.15)) - (1 - F) * alfa_1 * miu * A) * dt
        cout_NO3[1:-1] = c_NO3[1:-1] + adv_NO3 + dif_NO3 + reac_NO3 + ((S_NO3[1:-1] - c_NO3[1:-1])*caudales[1:-1, 1])

        adv_DQO = -((ki[2:] * v[2:] * c_DQO[2:] - ki[1:-1] * v[1:-1] * c_DQO[1:-1]) * (dt / dx) +
                    (kr[1:-1] * v[1:-1] * c_DQO[1:-1] - kr[0:-2] * v[0:-2] * c_DQO[0:-2]) * (dt / dx))
        dif_DQO = 0.5 * (d[2:] * c_DQO[2:] - 2 * d[1:-1] * c_DQO[1:-1] + d[0:-2] * c_DQO[0:-2]) * (dt / dx ** 2)
        reac_DQO = (-kDQO * c_DQO[0:-2] * p * (teta_DQO ** (c_T[0:-2] - 293.15))) * dt
        cout_DQO[1:-1] = c_DQO[1:-1] + adv_DQO + dif_DQO + reac_DQO + + ((S_DQO[1:-1] - c_DQO[1:-1])*caudales[1:-1, 1])

        adv_TDS = -((ki[2:] * v[2:] * c_TDS[2:] - ki[1:-1] * v[1:-1] * c_TDS[1:-1]) * (dt / dx) +
                    (kr[1:-1] * v[1:-1] * c_TDS[1:-1] - kr[0:-2] * v[0:-2] * c_TDS[0:-2]) * (dt / dx))
        dif_TDS = 0.5 * (d[2:] * c_TDS[2:] - 2 * d[1:-1] * c_TDS[1:-1] + d[0:-2] * c_TDS[0:-2]) * (dt / dx ** 2)
        reac_TDS = (-kTDS * c_TDS[0:-2]) * dt
        cout_TDS[1:-1] = c_TDS[1:-1] + adv_TDS + dif_TDS + reac_TDS + ((S_TDS[1:-1] - c_TDS[1:-1])*caudales[1:-1, 1])

        adv_EC = -((ki[2:] * v[2:] * c_EC[2:] - ki[1:-1] * v[1:-1] * c_EC[1:-1]) * (dt / dx) +
                   (kr[1:-1] * v[1:-1] * c_EC[1:-1] - kr[0:-2] * v[0:-2] * c_EC[0:-2]) * (dt / dx))
        dif_EC = 0.5 * (d[2:] * c_EC[2:] - 2 * d[1:-1] * c_EC[1:-1] + d[0:-2] * c_EC[0:-2]) * (dt / dx ** 2)
        reac_EC = (-kEC * c_EC[0:-2] * (teta_EC ** (c_T[0:-2] - 293.15))) * dt
        cout_EC[1:-1] = c_EC[1:-1] + adv_EC + dif_EC + reac_EC + ((S_EC[1:-1] - c_EC[1:-1])*caudales[1:-1, 1])

        adv_TC = -((ki[2:] * v[2:] * c_TC[2:] - ki[1:-1] * v[1:-1] * c_TC[1:-1]) * (dt / dx) +
                   (kr[1:-1] * v[1:-1] * c_TC[1:-1] - kr[0:-2] * v[0:-2] * c_TC[0:-2]) * (dt / dx))
        dif_TC = 0.5 * (d[2:] * c_TC[2:] - 2 * d[1:-1] * c_TC[1:-1] + d[0:-2] * c_TC[0:-2]) * (dt / dx ** 2)
        reac_TC = (-kTC * c_TC[0:-2] * (teta_TC ** (c_T[0:-2] - 293.15))) * dt
        cout_TC[1:-1] = c_TC[1:-1] + adv_TC + dif_TC + reac_TC + ((S_TC[1:-1] - c_TC[1:-1])*caudales[1:-1, 1])

        adv_GyA = -((ki[2:] * v[2:] * c_GyA[2:] - ki[1:-1] * v[1:-1] * c_GyA[1:-1]) * (dt / dx) +
                    (kr[1:-1] * v[1:-1] * c_GyA[1:-1] - kr[0:-2] * v[0:-2] * c_GyA[0:-2]) * (dt / dx))
        dif_GyA = 0.5 * (d[2:] * c_GyA[2:] - 2 * d[1:-1] * c_GyA[1:-1] + d[0:-2] * c_GyA[0:-2]) * (dt / dx ** 2)
        reac_GyA = (Jdbw / D + qtex / D - (kN + kH * c_pH[0:-2] - kOH * (Kw / c_pH[0:-2])) * fdw * c_GyA[0:-2] - kf * c_GyA[0:-2] - kb * c_GyA[
                                    0:-2] - kv * ((Cg / (Henry / (R * c_T[0:-2]))) - fdw * c_GyA[0:-2])) * dt/(10*tfactor)
        cout_GyA[1:-1] = c_GyA[1:-1] + adv_GyA + dif_GyA + reac_GyA + ((S_GyA[1:-1] - c_GyA[1:-1])*caudales[1:-1, 1])


        adv_Porg = -((ki[2:] * v[2:] * c_Porg[2:] - ki[1:-1] * v[1:-1] * c_Porg[1:-1]) * (dt / dx) +
                     (kr[1:-1] * v[1:-1] * c_Porg[1:-1] - kr[0:-2] * v[0:-2] * c_Porg[0:-2]) * (dt / dx))
        dif_Porg = 0.5 * (d[2:] * c_Porg[2:] - 2 * d[1:-1] * c_Porg[1:-1] + d[0:-2] * c_Porg[0:-2]) * (dt / dx ** 2)
        reac_Porg = (alfa_2 * resp * A - kPorg * c_Porg[0:-2] - kPsed * c_Porg[0:-2]) * dt
        cout_Porg[1:-1] = c_Porg[1:-1] + adv_Porg + dif_Porg + reac_Porg + ((S_Porg[1:-1] - c_Porg[1:-1])*caudales[1:-1, 1])

        adv_Pdis = -((ki[2:] * v[2:] * c_Pdis[2:] - ki[1:-1] * v[1:-1] * c_Pdis[1:-1]) * (dt / dx) +
                     (kr[1:-1] * v[1:-1] * c_Pdis[1:-1] - kr[0:-2] * v[0:-2] * c_Pdis[0:-2]) * (dt / dx))
        dif_Pdis = 0.5 * (d[2:] * c_Pdis[2:] - 2 * d[1:-1] * c_Pdis[1:-1] + d[0:-2] * c_Pdis[0:-2]) * (dt / dx ** 2)
        reac_Pdis = (kPorg * c_Porg[1:-1] + kPsed / D - sigma2 * miu * A) * dt
        cout_Pdis[1:-1] = c_Pdis[1:-1] + adv_Pdis + dif_Pdis + reac_Pdis + ((S_Pdis[1:-1] - c_Pdis[1:-1])*caudales[1:-1, 1])

        adv_TSS = -((ki[2:] * v[2:] * c_TSS[2:] - ki[1:-1] * v[1:-1] * c_TSS[1:-1]) * (dt / dx) +
                    (kr[1:-1] * v[1:-1] * c_TSS[1:-1] - kr[0:-2] * v[0:-2] * c_TSS[0:-2]) * (dt / dx))
        dif_TSS = 0.5 * (d[2:] * c_TSS[2:] - 2 * d[1:-1] * c_TSS[1:-1] + d[0:-2] * c_TSS[0:-2]) * (dt / dx ** 2)
        reac_TSS = qtex * (-Ws * c_TSS[0:-2] / D + Rs / D + Rp / D) * dt
        cout_TSS[1:-1] = c_TSS[1:-1] + adv_TSS + dif_TSS + reac_TSS + ((S_TSS[1:-1] - c_TSS[1:-1])*caudales[1:-1, 1])

        adv_SS = -((ki[2:] * v[2:] * c_SS[2:] - ki[1:-1] * v[1:-1] * c_SS[1:-1]) * (dt / dx) +
                   (kr[1:-1] * v[1:-1] * c_SS[1:-1] - kr[0:-2] * v[0:-2] * c_SS[0:-2]) * (dt / dx))
        dif_SS = 0.5 * (d[2:] * c_SS[2:] - 2 * d[1:-1] * c_SS[1:-1] + d[0:-2] * c_SS[0:-2]) * (dt / dx ** 2)
        reac_SS = qtex * (-Ws * c_SS[0:-2] / D + Rs / D + Rp / D) * dt
        cout_SS[1:-1] = c_SS[1:-1] + adv_SS + dif_SS + reac_SS + ((S_SS[1:-1] - c_SS[1:-1])*caudales[1:-1, 1])

        adv_ALK = -((ki[2:] * v[2:] * c_ALK[2:] - ki[1:-1] * v[1:-1] * c_ALK[1:-1]) * (dt / dx) + (
        kr[1:-1] * v[1:-1] * c_ALK[1:-1] - kr[0:-2] * v[0:-2] * c_ALK[0:-2]) * (dt / dx))
        dif_ALK = 0.5 * (d[2:] * c_ALK[2:] - 2 * d[1:-1] * c_ALK[1:-1] + d[0:-2] * c_ALK[0:-2]) * (dt / dx ** 2)
        reac_ALK = Wrp + Vv * As * (CO2S - ((c_pH[0:-2]) * (c_pH[0:-2]) / (((c_pH[0:-2]) * (c_pH[0:-2])) + K1 * (c_pH[0:-2]) + K1 * K2)) * c_ALK[0:-2])
        cout_ALK[1:-1] = c_ALK[1:-1] + adv_ALK + dif_ALK + reac_ALK + ((S_ALK[1:-1])*caudales[1:-1, 1])

        adv_pH = -((ki[2:] * v[2:] * c_pH[2:] - ki[1:-1] * v[1:-1] * c_pH[1:-1]) * (dt / dx) +
                   (kr[1:-1] * v[1:-1] * c_pH[1:-1] - kr[0:-2] * v[0:-2] * c_pH[0:-2]) * (dt / dx))
        dif_pH = 0.5 * (d[2:] * c_pH[2:] - 2 * d[1:-1] * c_pH[1:-1] + d[0:-2] * c_pH[0:-2]) * (dt / dx ** 2)
        reac_pH = ((Kw / (FrH * (c_ALK[0:-2])) ** 0.5))

        cout_pH[1:-1] = c_pH[1:-1] + adv_pH + dif_pH + reac_pH + ((S_pH[1:-1] - c_pH[1:-1])*caudales[1:-1, 1])

        c_T = cout_T
        c_OD = cout_OD
        c_DBO = cout_DBO
        c_NH3 = cout_NH3
        c_NO2 = cout_NO2
        c_NO3 = cout_NO3
        c_DQO = cout_DQO
        c_TDS = cout_TDS
        c_EC = cout_EC
        c_TC = cout_TC
        c_GyA = cout_GyA
        c_Porg = cout_Porg
        c_Pdis = cout_Pdis
        c_SS = cout_SS
        c_pH = cout_pH
        c_ALK = cout_ALK

        cout_T[-1] = cout_T[-2]
        cout_OD[-1] = cout_OD[-2]
        cout_DBO[-1] = cout_DBO[-2]
        cout_NH3[-1] = cout_NH3[-2]
        cout_OD[-1] = cout_OD[-2]
        cout_NO2[-1] = cout_NO2[-2]
        cout_NO3[-1] = cout_NO3[-2]
        cout_DQO[-1] = cout_DQO[-2]
        cout_TDS[-1] = cout_TDS[-2]
        cout_EC[-1] = cout_EC[-2]
        cout_TC[-1] = cout_TC[-2]
        cout_GyA[-1] = cout_GyA[-2]
        cout_Porg[-1] = cout_Porg[-2]
        cout_Pdis[-1] = cout_Pdis[-2]
        cout_SS[-1] = cout_SS[-2]
        cout_pH[-1] = cout_pH[-2]
        cout_ALK[-1] = cout_ALK[-2]

        cout_OD = np.where(cout_OD < 0, 0, cout_OD)
        cout_GyA = np.where(cout_GyA < 0, 0, cout_GyA)
        cout_NH3 = np.where(cout_NH3 < 0, 0, cout_NH3)
        cout_NO2 = np.where(cout_NO2 < 0, 0, cout_NO2)
        cout_NO3 = np.where(cout_NO3 < 0, 0, cout_NO3)
        cout_EC = np.where(cout_EC < 0, 0, cout_EC)
        cout_TC = np.where(cout_TC < 0, 0, cout_TC)

    return c_T, c_OD, c_DBO, c_NH3, c_NO2, c_NO3, c_DQO, c_TDS, c_EC, c_TC, c_GyA, c_Porg, c_Pdis, c_TSS, c_SS, c_pH, c_ALK, dt

def run(arhivo_entrada, tiempo, directorio_salida, variables, show, export):
    # Numero de pasos en el tiempo a ejecutar
    nt = tiempo
    ct = (np.arange(1, nt))

    # Reading input data from Excel file
    # xls_config = "Rio_Los_Ranchos_prueba_00.xlsx"
    hmed, slope, vel, b_c, i_c, ST, SOD, SDBO, SNH3, SNO2, SNO3, STDS, SGyA, SDQO, SPorg, SPdis, SEC, STC, STSS, SSS, SpH, SALK, Caudales = read_config_file(arhivo_entrada)

    #Discretizacion en el espacio
    dx = hmed[1, 0] - hmed[0, 0]
    # velocidad del agua en cada punto de monitoreo
    va = vel[:, 1]
    # coeficiente de difusión en cada punto de monitoreo
    Diff = variables['Diff']

    cd = np.zeros(len(va)) + Diff
    v = np.zeros(len(va)) + np.mean(vel)
    D = np.mean(hmed)

    # Condiciones de Frontera
    # TEMPERATURA
    b_c_T = b_c[:, 14]
    b_c_T = b_c_T + 273.15
    # OXIGENO DISUELTO
    b_c_OD = b_c[:, 1]
    # DBO
    b_c_DBO = b_c[:, 2]
    # Amonio
    b_c_NH3 = b_c[:, 3]
    # Nitritos
    b_c_NO2 = b_c[:, 4]
    # Nitratos
    b_c_NO3 = b_c[:, 5]
    # DQO
    b_c_DQO = b_c[:, 9]
    # TDS
    b_c_TDS = b_c[:, 6]
    kcondt = 1.92
    # EC
    b_c_EC = b_c[:, 12]
    # TC
    b_c_TC = b_c[:, 13]
    # GyA
    b_c_GyA = b_c[:, 7]
    # P organico
    b_c_Porg = b_c[:, 10]
    # P disuelto
    b_c_Pdis = b_c[:, 11]
    # Solidos suspendidos
    b_c_TSS = b_c[:, 15]
    # Solidos sedimentables
    b_c_SS = b_c[:, 16]
    # pH
    b_c_pH = b_c[:, 17]
    b_c_pH = 10 ** (-1 * (b_c_pH))
    # Alkalinidad
    b_c_ALK = b_c[:, 18]

    # Condiciones Iniciales
    # TEMPERATURA
    i_c_T = i_c[:, 14]
    i_c_T = i_c_T + 273.15
    # OXIGENO DISUELTO
    i_c_OD = i_c[:, 1]
    # DBO
    i_c_DBO = i_c[:, 2]
    # Amonio
    i_c_NH3 = i_c[:, 3]
    # Nitritos
    i_c_NO2 = i_c[:, 4]
    # Nitratos
    i_c_NO3 = i_c[:, 5]
    # DQO
    i_c_DQO = i_c[:, 9]
    # TDS
    i_c_TDS = i_c[:, 6]
    # EC
    i_c_EC = i_c[:, 12]
    # TC
    i_c_TC = i_c[:, 13]
    # GyA
    i_c_GyA = i_c[:, 7]
    # P organico
    i_c_Porg = i_c[:, 10]
    # P disuelto
    i_c_Pdis = i_c[:, 11]
    # Solidos suspendidos
    i_c_TSS = i_c[:, 15]
    # Solidos sedimentables
    i_c_SS = i_c[:, 16]
    # pH
    i_c_pH = i_c[:, 17]
    i_c_pH = 10 ** (-1 * (i_c_pH))
    # Alkalinidad
    i_c_ALK = i_c[:, 18]

    mconT = np.empty((nt, np.size(i_c_T, axis=0)))
    mconT[0, :] = i_c_T
    mconOD = np.empty((nt, np.size(i_c_OD, axis=0)))
    mconOD[0, :] = i_c_OD
    mconDBO = np.empty((nt, np.size(i_c_DBO, axis=0)))
    mconDBO[0, :] = i_c_DBO
    mconNH3 = np.empty((nt, np.size(i_c_NH3, axis=0)))
    mconNH3[0, :] = i_c_NH3
    mconNO2 = np.empty((nt, np.size(i_c_NO2, axis=0)))
    mconNO2[0, :] = i_c_NO2
    mconNO3 = np.empty((nt, np.size(i_c_NO3, axis=0)))
    mconNO3[0, :] = i_c_NO3
    mconDQO = np.empty((nt, np.size(i_c_DQO, axis=0)))
    mconDQO[0, :] = i_c_DQO
    mconTDS = np.empty((nt, np.size(i_c_TDS, axis=0)))
    mconTDS[0, :] = i_c_TDS
    mconEC = np.empty((nt, np.size(i_c_EC, axis=0)))
    mconEC[0, :] = i_c_EC
    mconTC = np.empty((nt, np.size(i_c_TC, axis=0)))
    mconTC[0, :] = i_c_TC
    mconGyA = np.empty((nt, np.size(i_c_GyA, axis=0)))
    mconGyA[0, :] = i_c_GyA
    mconPorg = np.empty((nt, np.size(i_c_Porg, axis=0)))
    mconPorg[0, :] = i_c_Porg
    mconPdis = np.empty((nt, np.size(i_c_Pdis, axis=0)))
    mconPdis[0, :] = i_c_Pdis
    mconTSS = np.empty((nt, np.size(i_c_TSS, axis=0)))
    mconTSS[0, :] = i_c_TSS
    mconSS = np.empty((nt, np.size(i_c_SS, axis=0)))
    mconSS[0, :] = i_c_SS
    mconpH = np.empty((nt, np.size(i_c_pH, axis=0)))
    mconpH[0, :] = i_c_pH
    mconALK = np.empty((nt, np.size(i_c_ALK, axis=0)))
    mconALK[0, :] = i_c_ALK

    ST = ST[:, 1:]
    SOD = SOD[:, 1:]
    SDBO = SDBO[:, 1:]
    SNH3 = SNH3[:, 1:]
    SNO2 = SNO2[:, 1:]
    SNO3 = SNO3[:, 1:]
    STDS = STDS[:, 1:]
    SGyA = SGyA[:, 1:]
    SDQO = SDQO[:, 1:]
    SPorg = SPorg[:, 1:]
    SPdis = SPdis[:, 1:]
    SEC = SEC[:, 1:]
    STC = STC[:, 1:]
    STSS = STSS[:, 1:]
    SSS = SSS[:, 1:]
    SpH = SpH[:, 1:]
    SALK = SALK[:, 1:]

    for i in range(1, nt):
        muestra = int(i / 3600)
        i_c_T[0] = b_c_T[muestra]
        i_c_OD[0] = b_c_OD[muestra]
        i_c_DBO[0] = b_c_DBO[muestra]
        i_c_NH3[0] = b_c_NH3[muestra]
        i_c_NO2[0] = b_c_NO2[muestra]
        i_c_NO3[0] = b_c_NO3[muestra]
        i_c_DQO[0] = b_c_DQO[muestra]
        i_c_TDS[0] = b_c_TDS[muestra]
        i_c_EC[0] = b_c_EC[muestra]
        i_c_TC[0] = b_c_TC[muestra]
        i_c_GyA[0] = b_c_GyA[muestra]
        i_c_Porg[0] = b_c_Porg[muestra]
        i_c_Pdis[0] = b_c_Pdis[muestra]
        i_c_TSS[0] = b_c_TSS[muestra]
        i_c_SS[0] = b_c_SS[muestra]
        i_c_pH[0] = b_c_pH[muestra]
        i_c_ALK[0] = b_c_ALK[muestra]

        S_T = ST[:, muestra]
        S_OD = SOD[:, muestra]
        S_DBO = SDBO[:, muestra]
        S_NH3 = SNH3[:, muestra]
        S_NO2 = SNO2[:, muestra]
        S_NO3 = SNO3[:, muestra]
        S_TDS = STDS[:, muestra]
        S_GyA = SGyA[:, muestra]
        S_DQO = SDQO[:, muestra]
        S_Porg = SPorg[:, muestra]
        S_Pdis = SPdis[:, muestra]
        S_EC = SEC[:, muestra]
        S_TC = STC[:, muestra]
        S_TSS = STSS[:, muestra]
        S_SS = SSS[:, muestra]
        S_pH = SpH[:, muestra]
        S_ALK = SALK[:, muestra]

        #  Evolución de la concentración para t + dt
        T, OD, DBO, NH3, NO2, NO3, DQO, TDS, EC, TC, GyA, Porg, Pdis, TSS, SS, pH, ALK, paso_t = calidad_explicito(D, dx, i_c_T,
            i_c_OD, i_c_DBO, i_c_NH3, i_c_NO2, i_c_NO3, i_c_DQO, i_c_TDS, i_c_EC, i_c_TC, i_c_GyA,i_c_Porg, i_c_Pdis, i_c_TSS, i_c_SS,
            i_c_pH, i_c_ALK, v, cd, S_T, S_OD, S_DBO, S_NH3, S_NO2, S_NO3, S_DQO, S_TDS, S_EC, S_TC, S_GyA, S_Porg, S_Pdis, S_TSS, S_SS,
            S_pH, S_ALK, Caudales, variables)

        # Se guardan las concentraciones del momento t+dt
        mconT[i, :] = T
        mconOD[i, :] = OD
        mconDBO[i, :] = DBO
        mconNH3[i, :] = NH3
        mconNO2[i, :] = NO2
        mconNO3[i, :] = NO3
        mconDQO[i, :] = DQO
        mconTDS[i, :] = TDS
        mconEC[i, :] = EC
        mconTC[i, :] = TC
        mconGyA[i, :] = GyA
        mconPorg[i, :] = Porg
        mconPdis[i, :] = Pdis
        mconTSS[i, :] = TSS
        mconSS[i, :] = SS
        mconpH[i, :] = pH
        mconALK[i, :] = ALK

        # Actualizar condición inicial
        i_c_T = T
        i_c_OD = OD
        i_c_DBO = DBO
        i_c_NH3 = NH3
        i_c_NO2 = NO2
        i_c_NO3 = NO3
        i_c_DQO = DQO
        i_c_TDS = TDS
        i_c_EC = EC
        i_c_TC = TC
        i_c_GyA = GyA
        i_c_Porg = Porg
        i_c_Pdis = Pdis
        i_c_TSS = TSS
        i_c_SS = SS
        i_c_pH = pH
        i_c_ALK = ALK
        print(ALK)
        paso_de_tiempo = paso_t

    mconConduct = kcondt * mconTDS
    mconT = mconT - 273.15
    mconpH = np.where(mconpH < 0.000000001, 0.000000001, mconpH)
    mconpH = (np.log10(mconpH))*(-1)
    pH = (np.log10(pH))*(-1)
    # fix_print_with_import
    print("Guardando datos de salida...")

    book = xlwt.Workbook()
    save_sheet(book, 'T', mconT[0::3600, :])
    save_sheet(book, 'OD', mconOD[0::3600, :])
    save_sheet(book, 'DBO', mconDBO[0::3600, :])
    save_sheet(book, 'NH3', mconNH3[0::3600, :])
    save_sheet(book, 'NO2', mconNO2[0::3600, :])
    save_sheet(book, 'NO3', mconNO3[0::3600, :])
    save_sheet(book, 'DQO', mconDQO[0::3600, :])
    save_sheet(book, 'TDS', mconTDS[0::3600, :])
    save_sheet(book, 'EC', mconEC[0::3600, :])
    save_sheet(book, 'TC', mconTC[0::3600, :])
    save_sheet(book, 'GyA', mconGyA[0::3600, :])
    save_sheet(book, 'Conduct', mconConduct[0::3600, :])
    save_sheet(book, 'Porg', mconPorg[0::3600, :])
    save_sheet(book, 'Pdis', mconPdis[0::3600, :])
    save_sheet(book, 'TSS', mconTSS[0::3600, :])
    save_sheet(book, 'SS', mconSS[0::3600, :])
    save_sheet(book, 'pH', mconpH[0::3600, :])
    save_sheet(book, 'ALK', mconALK[0::3600, :])

    used_vars(book, variables)
    book.save(join(directorio_salida, "Resultados.xls"))

    if show:

        # fix_print_with_import
        print(u"Creando Graficas")

        #Graficas en el tiempo
        xlabel = 'Tiempo(s)'
        ylabel = '(mg/L)'
        x_data = ct[1::3600]

        fig, ax = plt.subplots(5, 3, sharex=True)
        fig.add_subplot("111", frameon=False)
        fig.canvas.set_window_title('Graficas de Tiempo.')

        plot(ax[0,1], 'Evalucion OD en punto final', [x_data, mconOD[1::3600, -1]])
        plot(ax[0,2], 'Evalucion DBO en punto final', [x_data, mconDBO[1::3600, -1]])
        plot(ax[1,0], 'Evalucion NH3 en punto final', [x_data, mconNH3[1::3600, -1]])
        plot(ax[1,1], 'Evalucion NO2 en punto final', [x_data, mconNO2[1::3600, -1]])
        plot(ax[1,2], 'Evalucion NO3 en punto final', [x_data, mconNO3[1::3600, -1]])
        plot(ax[2,0], 'Evalucion DQO en punto final', [x_data, mconDQO[1::3600, -1]])
        plot(ax[2,1], 'Evalucion TDS en punto final', [x_data, mconTDS[1::3600, -1]])
        plot(ax[2,2], 'Evalucion EC en punto final', [x_data, mconEC[1::3600, -1]])
        plot(ax[3,0], 'Evalucion TC en punto final', [x_data, mconTC[1::3600, -1]])
        plot(ax[3,1], 'Evalucion Grasas y Aceites en punto final', [x_data, mconGyA[1::3600, -1]])
        plot(ax[3,2], 'Evalucion P org en punto final', [x_data, mconPorg[1::3600, -1]])
        plot(ax[4,0], 'Evalucion P disuelto en punto final', [x_data, mconPdis[1::3600, -1]])
        plot(ax[4,1], 'Evalucion del pH en punto final', [x_data, mconpH[1::3600, -1]])
        plot(ax[4,2], 'Evalucion Alcanilidad en punto final', [x_data, mconALK[1::3600, -1]])

        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.subplots_adjust(hspace=0.5)
        plt.grid(False)
        plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
        plt.xlabel(xlabel, fontsize=15)
        plt.ylabel(ylabel, fontsize=15)
        fig.show()

        #Graficas en el espacio
        xlabel = 'Distancia(m)'
        ylabel = 'Concentracion (mg/L)'
        c_x = hmed[:, 0]

        fig2, ax2 = plt.subplots(5, 3, sharex=True)
        fig2.add_subplot(111, frameon=False)
        fig2.canvas.set_window_title('Graficas de espacio.')

        plot(ax2[0,1], 'Evalucion OD en el espacio', [c_x, OD])
        plot(ax2[0,2], 'Evalucion DBO en el espacio', [c_x, DBO])
        plot(ax2[1,0], 'Evalucion NH3 en el espacio', [c_x, NH3])
        plot(ax2[1,1], 'Evalucion NO2 en el espacio', [c_x, NO2])
        plot(ax2[1,2], 'Evalucion NO3 en el espacio', [c_x, NO3])
        plot(ax2[2,0], 'Evalucion DQO en el espacio', [c_x, DQO])
        plot(ax2[2,1], 'Evalucion TDS en el espacio', [c_x, TDS])
        plot(ax2[2,2], 'Evalucion EC en el espacio', [c_x, EC])
        plot(ax2[3,0], 'Evalucion TC en el espacio', [c_x, TC])
        plot(ax2[3,1], 'Evalucion Grasas y Aceites en el espacio', [c_x, GyA])
        plot(ax2[3,2], 'Evalucion P organico en el espacio', [c_x, Porg])
        plot(ax2[4,0], 'Evalucion P disuelto en el espacio', [c_x, Pdis])
        plot(ax2[4,1], 'Evalucion pH en espacio', [c_x, pH])
        plot(ax2[4,2], 'Evalucion Alcanilidad en espacio', [c_x, ALK])

        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.subplots_adjust(hspace=0.5)
        plt.grid(False)
        plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useOffset=True)
        plt.xlabel(xlabel, fontsize=15)
        plt.ylabel(ylabel, fontsize=15)
        fig2.show()


    if export:

        # fix_print_with_import
        print(u"Guardando Graficas...")

        xlabel = 'Tiempo(s)'
        ylabel = 'Concentracion (mg/L)'
        x_data = ct[1::3600]

        plt.figure('tmp')
        # Gráficas de tiempo
        save_plot(plt, 'Evalucion T en punto final en tiempo', xlabel, ylabel, [x_data, mconT[1::3600, -1]], directorio_salida)
        save_plot(plt, 'Evalucion OD en punto final en tiempo', xlabel, ylabel, [x_data, mconOD[1::3600, -1]], directorio_salida)
        save_plot(plt, 'Evalucion DBO en punto final en tiempo', xlabel, ylabel, [x_data, mconDBO[1::3600, -1]], directorio_salida)
        save_plot(plt, 'Evalucion NH3 en punto final en tiempo', xlabel, ylabel, [x_data, mconNH3[1::3600, -1]], directorio_salida)
        save_plot(plt, 'Evalucion NO2 en punto final en tiempo', xlabel, ylabel, [x_data, mconNO2[1::3600, -1]], directorio_salida)
        save_plot(plt, 'Evalucion NO3 en punto final en tiempo', xlabel, ylabel, [x_data, mconNO3[1::3600, -1]], directorio_salida)
        save_plot(plt, 'Evalucion DQO en punto final en tiempo', xlabel, ylabel, [x_data, mconDQO[1::3600, -1]], directorio_salida)
        save_plot(plt, 'Evalucion TDS en punto final en tiempo', xlabel, ylabel, [x_data, mconTDS[1::3600, -1]], directorio_salida)
        save_plot(plt, 'Evalucion EC en punto final en tiempo', xlabel, ylabel, [x_data, mconEC[1::3600, -1]], directorio_salida)
        save_plot(plt, 'Evalucion TC en punto final en tiempo', xlabel, ylabel, [x_data, mconTC[1::3600, -1]], directorio_salida)
        save_plot(plt, 'Evalucion Grasas y Aceites en punto final en tiempo', xlabel, ylabel, [x_data, mconGyA[1::3600, -1]], directorio_salida)
        save_plot(plt, 'Evalucion P org en punto final en tiempo', xlabel, ylabel, [x_data, mconPorg[1::3600, -1]], directorio_salida)
        save_plot(plt, 'Evalucion P disuelto en punto final en tiempo', xlabel, ylabel, [x_data, mconPdis[1::3600, -1]], directorio_salida)
        save_plot(plt, 'Evalucion del pH en punto final', xlabel, 'pH', [x_data, mconpH[1::3600, -1]], directorio_salida)
        save_plot(plt, 'Evalucion Alcanilidad en punto final', xlabel, 'CaCO3/L', [x_data, mconALK[1::3600, -1]], directorio_salida)

        # Gráficas de espacio
        xlabel = 'Distancia(m)'
        ylabel = 'Concentracion (mg/L)'
        c_x = hmed[:, 0]

        save_plot(plt, 'Evalucion T en el espacio',xlabel, ylabel, [c_x, T], directorio_salida)
        save_plot(plt, 'Evalucion OD en el espacio',xlabel, ylabel, [c_x, OD], directorio_salida)
        save_plot(plt, 'Evalucion DBO en el espacio',xlabel, ylabel, [c_x, DBO], directorio_salida)
        save_plot(plt, 'Evalucion NH3 en el espacio',xlabel, ylabel, [c_x, NH3], directorio_salida)
        save_plot(plt, 'Evalucion NO2 en el espacio',xlabel, ylabel, [c_x, NO2], directorio_salida)
        save_plot(plt, 'Evalucion NO3 en el espacio',xlabel, ylabel, [c_x, NO3], directorio_salida)
        save_plot(plt, 'Evalucion DQO en el espacio',xlabel, ylabel, [c_x, DQO], directorio_salida)
        save_plot(plt, 'Evalucion TDS en el espacio',xlabel, ylabel, [c_x, TDS], directorio_salida)
        save_plot(plt, 'Evalucion EC en el espacio',xlabel, ylabel, [c_x, EC], directorio_salida)
        save_plot(plt, 'Evalucion TC en el espacio',xlabel, ylabel, [c_x, TC], directorio_salida)
        save_plot(plt, 'Evalucion Grasas y Aceites en el espacio',xlabel, ylabel, [c_x, GyA], directorio_salida)
        save_plot(plt, 'Evalucion P organico en el espacio',xlabel, ylabel, [c_x, Porg], directorio_salida)
        save_plot(plt, 'Evalucion P disuelto en el espacio',xlabel, ylabel, [c_x, Pdis], directorio_salida)
        save_plot(plt, 'Evalucion del pH en espacio',xlabel, 'pH', [c_x, pH], directorio_salida)
        save_plot(plt, 'Evalucion Alcanilidad en espacio',xlabel, 'CaCO3', [c_x, ALK], directorio_salida)

    # fix_print_with_import
    print("El proceso ha finalizado.")


if __name__ == '__main__':
    archivo_entrada = 'Prueba_CAR.xls'
    tiempo = 86400
    directorio_salida = './salida/'
    show = False
    export = True

    variables = {'Da': 1.6296e-07, 'ko2': 0.0002787, 'cs': 8.0, 'knh3': 5.787e-05, 'ksnh3': 1.15741e-06,
        'alfa_nh3': 2.0, 'kdbo': 1.1574e-06, 'ks': 0.4, 'alfa_no2': 1.1, 'ksod': 1.15e-06,
        'knt': 2.3148e-06, 'NT': 0.5, 'kno2': 1.1574e-05, 'kno3': 1.1574e-06, 'kDQO': 1.1574e-06,
        'kTDS': 2e-08, 'A': 0.001, 'alfa_1': 0.175, 'miu': 2.31481e-05, 'F': 0.1, 'kTC': 2.31481e-06,
        'teta_TC': 0.9, 'kEC': 6.31481e-06, 'teta_EC': 0.8, 'Jdbw': 4.62963e-08, 'qtex': 3.47222e-05,
        'kN': 1.1574e-07, 'kH': 1.1574e-05, 'kOH': 8.64e-10, 'fdw': 0.5, 'kf': 1.1574e-07,
        'kb': 1.1574e-08, 'kv': 1.1574e-08, 'Cg': 1e-05, 'Henry': 0.001, 'R': 8.205746e-05,'T': 295.15,
        'alfa_2': 0.5, 'resp': 0.25, 'kPorg': 1.1574e-06, 'kPsed': 1.574e-06, 'sigma2': 8.587e-05,
        'Ws': 0.05, 'Rs': 0.1, 'Rp': 0.1, 'k': 0.58, 'den': 997.3, 'Cp': 4148.1, 'teta_DBO': 1.01,
        'teta_NH3': 1.01, 'teta_NO2': 1.01, 'teta_DQO': 1.01, 'teta_NT': 1.01, 'teta_NO3': 1.01,
        'Kw': 1e-14, 'K1': 4.5e-07, 'K2': 4.7e-11, 'Vv': 5.787e-06, 'As': 1.0, 'CO2S': 1.23e-05,
        'Wrp': 1.808e-09, 'FrH': 0.0172, 'Diff': 5.0, 'As1': 1.0, 'Jsn': 145.0, 'sbc': 5.67e-08,
        'Tair': 17.0, 'Aair': 0.6, 'eair': 14.3, 'RL': 0.03, 'Uw': 3.0, 'es': 11.5, 'tfactor': 1.0}

    run(archivo_entrada, tiempo, directorio_salida, variables, show, export)
