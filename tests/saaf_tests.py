from itertools import product

from nose.tools import *
import numpy as np
from numpy.linalg import cholesky

import material
import mesh
from mesh import Cell
from saaf import SAAF
class TestSaaf:
    # Tests to verify SAAF is working

    @classmethod

    def setup_class(cls):
        #Specify problem here:
        from input_void import problem

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

        cls.saaf = SAAF(mat_cls=MAT_LIB, mesh_cls=MESH, prob_dict=problem)
        cls.LHS = cls.saaf.assemble_bilinear_forms()

    def test_bilinear_symmetric(self):
        # Checks that SAAF LHS is symmetric
        for gd in self.saaf._group_dir_pairs():
            ok_((self.LHS[gd].transpose() != self.LHS[gd]).nnz == 0, 
                "Bilinear forms should be symmetric")

    def test_bilinear_positive_definite(self):
        # Checks that SAAF LHS is positive definite
        for gd in self.saaf._group_dir_pairs():
            cholesky(self.LHS[gd].todense())





