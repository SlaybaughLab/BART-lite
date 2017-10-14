from os import listdir
from os.path import isfile, join

mat_dir = './mat/kaist/' # Location of material files
mats = [mat_dir + f for
        f in listdir(mat_dir) if isfile(join(mat_dir, f))]

layout_mox1 = """
  43 43 43 43 43 43 43 43 43 43 43 43 43 43 43 43 43
  43 43 43 43 43 43 43 43 43 43 43 43 43 43 43 43 43
  70 70 70 70 70 70 70 70 70 70 70 70 70 43 43 43 43
  70 70 70 70 70 70 70 70 70 70 70 70 70 43 43 43 43
  gt 70 70 70 70 gt gt 70 70 70 70 70 70 70 70 43 43
  gt 70 70 70 70 gt gt 70 70 70 70 70 70 70 70 43 43
  70 70 70 70 70 70 70 70 70 gt gt 70 70 70 70 43 43
  70 70 70 70 70 70 70 70 70 gt gt 70 70 70 70 43 43
  87 87 87 87 87 87 87 aa aa 70 70 70 70 70 70 43 43
  87 87 87 87 87 87 87 aa aa 70 70 70 70 70 70 43 43
  gt 87 87 87 87 gt gt 87 87 70 70 gt gt 70 70 43 43
  gt 87 87 87 87 gt gt 87 87 70 70 gt gt 70 70 43 43
  87 87 87 bb bb 87 87 87 87 70 70 70 70 70 70 43 43
  87 87 87 bb bb 87 87 87 87 70 70 70 70 70 70 43 43
  87 87 87 87 87 87 87 87 87 70 70 70 70 70 70 43 43
  87 87 87 87 87 87 87 87 87 70 70 70 70 70 70 43 43
  gt 87 87 87 87 gt gt 87 87 70 70 gt gt 70 70 43 43
"""
layout_dict={'43': 'mox_43',
             '70': 'mox_70',
             '87': 'mox_87',
             'gt': 'guide_tube',
             'aa': 'mox_70',
             'bb': 'mox_87'}

problem = {
    "sn_order": 2,              # REQ: SN angular quadrature order
    "do_nda": False,            # REQ: to determine whether or not to use NDA
    "do_ua": False,            # REQ: to determine use UA for NDA or not
    "mesh_cells": 17,           # REQ: number of cells per side
    "groups": 7,                # REQ: number of energy groups
    "domain_upper": 10,         # REQ: domain size
    "materials": mats,          # REQ: list of xml material files
    "layout": layout_mox1,      # REQ: material layout to use
    "layout_dict": layout_dict, # REQ: material layout dictionary
    "tr_scatt": True            # OP:  Take transp. of scatt. matrices
}
