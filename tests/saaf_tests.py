from itertools import product

from nose.tools import *
import numpy as np

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

    def test_bilinear_symmetric(self):
        # Check that LHS is symmetric
        LHS = self.saaf.assemble_bilinear_forms()
        ok_((LHS[g].transpose() == LHS[g] for g in xrange(len(LHS))), 
           "Bilinear forms should be symmetric")



