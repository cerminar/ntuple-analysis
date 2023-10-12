# run with python -m python.plotters_config
from __future__ import absolute_import
from . import plotters
from . import collections
from . import selections

from cfg import *


if __name__ == "__main__":

    print('enter name: ')
    selec_name = input()
    sel_list = []
    sel_list = eval(selec_name)
    for sel in sel_list:
        print(sel)
