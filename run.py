from __future__ import division
from input import problem
from mesh import mesh_gen
import matplotlib.pyplot as plt
import build_cells
import nda
import materials

# Geometry Setup
DOMAIN_LOWER = build_cells.DOMAIN_LOWER
DOMAIN_UPPER = build_cells.DOMAIN_UPPER

CELLS = problem["mesh cells"]
DOMAIN_LENGTH = DOMAIN_UPPER - DOMAIN_LOWER
CELL_LENGTH = DOMAIN_LENGTH/CELLS

DATA = build_cells.cell_to_metadata

# Materials Setup
NUM_GRPS = problem["groups"]
matlibs = []
# matmaps = TODO: set up matmap
for mats in problem["material files"]:
  matlibs.append(materials.mat_lib(NUM_GRPS, files=mats))



def main():
  u = fempoi2d.fempoi2d(CELL_LENGTH, DOMAIN_LENGTH, DATA)
  x = mesh_gen(CELLS)[:, :, 0]
  y = mesh_gen(CELLS)[:, :, 1]
  cset1 = plt.contourf(x, y, u, 10)
  plt.colorbar()
  plt.contour(x, y, u, cset1.levels, hold='on', colors='k')
  plt.axis('equal')
  plt.show()

if __name__ == '__main__':
  main()


