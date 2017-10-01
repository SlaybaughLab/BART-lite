from __future__ import division
from input import problem
from collections import namedtuple
from itertools import product
import numpy as np

# Set domain boundaries, assuming square domain
DOMAIN_LOWER = 0
DOMAIN_UPPER = 1

# Import number of cells from input
CELLS = problem["mesh_cells"]

# Calculate domain and cell lengths
DOMAIN_LENGTH = DOMAIN_UPPER - DOMAIN_LOWER
CELL_LENGTH = DOMAIN_LENGTH/CELLS

# Initialize Cells, MetaData, and Coordinates
Cell=namedtuple("Cell", ["i", "j"]) 
MetaData=namedtuple("MetaData", ["mat_id", "coordinates", "cross_sections"])
Coordinates=namedtuple("Coordinates", ["x_lower", "x_upper", "y_lower", "y_upper"])

# Name cells based on cell indices
i = np.repeat(np.arange(CELLS), CELLS)
j = np.tile(np.arange(CELLS), CELLS)
index = zip(i, j)
cells = [Cell(x,y) for x,y in index]

# Converts indicies to coordinates
def index_to_coordinates(cell):
    x_lower = CELL_LENGTH*cell.i
    x_upper = x_lower + CELL_LENGTH
    y_lower = CELL_LENGTH*cell.j
    y_upper = y_lower + CELL_LENGTH
    return Coordinates(x_lower, x_upper, y_lower, y_upper)

def metadata(cell):
  return MetaData(None, index_to_coordinates(cell), None)
  # Note: Has no mat id or cross section info. TODO. 

cell_to_metadata = {c:metadata(c) for c in cells}
