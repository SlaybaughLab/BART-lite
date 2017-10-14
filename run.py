from __future__ import division
import matplotlib.pyplot as plt
import material
import mesh

# iterations
from eigen_iterations import Eigen
from nda import NDA
from saaf import SAAF

#Specify problem here:
from input_kaist_mox_1 import problem

# Build Material Library

try:
    MAT_LIB = material.mat_lib(
        n_grps=problem['groups'],
        files=problem['materials'],
        tr_scatt=problem['tr_scatt'])
except KeyError:
    MAT_LIB = material.mat_lib(
        n_grps=problem['groups'], files=problem['materials'])

# Build Material Mapping

MAT_MAP = material.mat_map(
    lib=MAT_LIB,
    layout=problem['layout'],
    layout_dict=problem['layout_dict'],
    x_max=problem['domain_upper'],
    n=problem['mesh_cells'])

# Build Mesh

MESH = mesh.Mesh(problem['mesh_cells'], problem['domain_upper'], MAT_MAP)


def run():
    # do we do NDA
    do_nda = problem['do_nda']
    # Eigen class construction
    eigen_cls = Eigen()
    # construct HO solver
    ho_cls = SAAF(mat_cls=MAT_LIB, mesh_cls=MESH, prob_dict=problem)
    if not do_nda:
        # eigen solving
        eigen_cls.do_iterations(ho_cls=ho_cls, nda_cls=None)
    else:
        # construct NDA solver
        nda_cls = NDA(mat_cls=MAT_LIB, mesh_cls=MESH, prob_dict=problem)
        # eigen solving
        eigen_cls.do_iterations(ho_cls=ho_cls, nda_cls=nda_cls)
    # TODO: output and plotting functionality


run()
