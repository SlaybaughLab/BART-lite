from __future__ import division
from input import problem
from mesh import mesh_gen
import matplotlib.pyplot as plt
import build_cells
import fempoi2d

DOMAIN_LOWER = build_cells.DOMAIN_LOWER
DOMAIN_UPPER = build_cells.DOMAIN_UPPER

CELLS = problem["mesh_cells"]
DOMAIN_LENGTH = DOMAIN_UPPER - DOMAIN_LOWER
CELL_LENGTH = DOMAIN_LENGTH/CELLS

DATA = build_cells.cell_to_metadata

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


