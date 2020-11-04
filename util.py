from os import path

import xlwt
import numpy as np

BOLD_FONT_XLWT = xlwt.Style.easyxf('font: bold on;')


def join(folder, name):
    return path.join(folder, name)


def used_vars(book, variables):
    sheet = book.add_sheet('Variables Usadas')

    sheet.write(0, 0, 'Variables Usadas', BOLD_FONT_XLWT)
    idx = 1

    for k, v in variables.items():
        sheet.write(idx, 0, k, BOLD_FONT_XLWT)
        sheet.write(idx, 1, v)

        idx += 1


def read_sheet(workbook, name):
    """
    Reads the sheet of the given workbook and returns a numpy.ndarray
    :param workbook: File that has the information.
    :param name: Name of the sheet
    :return: numpy.ndarray
    """

    sheet = workbook.sheet_by_name(name)

    # Number of written Rows in sheet
    r = sheet.nrows
    # Number of written Columns in sheet
    c = sheet.ncols

    data = np.zeros([r - 1, c])
    # Reading each cell in excel sheet 'BC'
    for i in range(1, r):
        for j in range(c):
            data[i - 1, j] = float(sheet.cell_value(i, j))

    return data


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
            sheet.write(i, j, data[i - 1, j - 1])
