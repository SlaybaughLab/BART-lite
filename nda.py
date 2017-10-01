import numpy as np
import scipy.sparse as sps

class NDA(object):
  def __init__(self, mat):
    """ @brief constructor of NDA class

    @param mat of type material.material. """
    
    self.mat = mat # materials library
    self.num_materials = # extract number of materials 
    self.n_grp = #
    # matrices and vectors
    self.sys_mats = [0 for _ in xrange(self.n_grp)]
    self.sys_rhses = 

  def assemble_fixed_linear_forms(self, sflxes_prev=None):
    '''@brief a function used to assemble fixed source or fission source on the
    rhs for all components
    Generate numpy arrays and put them in self.
    '''
    # Loop over groups
    f = sps.lil_matrix((nodes, 1))
    q = sps.lil_matrix((nodes, 1))
    j = sps.lil_matrix((nodes, 1))

    elem_fs = [] # list of elementary matrices based on material
    elem_qs = []
    elem_js = []

    for m in mat.ids():
      # Set up elementary matrices for each material

      # TODO: Retrieve cross-sections for cell
      sigf = self.mat.get('sig_t', mat_id=m)[g] # material and group check with josh
      nu = self.mat.get('nu', mad_id=m)
        
      # TODO: Calculate j
      j = 1

      # Assemble LHS
      area = h**2 # retrieve h
      perimeter = 4*h
      elem_f = np.zeros(4)
      elem_q = np.zeros(4)
      elem_j = np.zeros(4)

      for ii in xrange(4):
        elem_f[ii] = nu*sigf*sflxes_prev[g]*area
        elem_q[ii] = q*area
        elem_j[ii] = 2*j*perimeter

      elem_fs.append(elem_f)
      elem_qs.append(elem_q)
      elem_js.append(elem_j)

    for cell in xrange(n):
      cell_i = cell//int(np.sqrt(n))
      cell_j = cell%int(np.sqrt(n))

      m = # retrieve material id

      # Global to Local node mapping
      """Nodes are flattened by row with row x=0 going first.
      That means node x0,y0 in cell0 has a global index of 0
      and node x1,y1 in cell0 has a global index of nodes_x+1"""
      mapping = np.array([cell+cell_i, cell+cell_i+1, 
                            cell+cell_i+n_x+1, cell+cell_i+n_x+2])
      xx = 0
      for ii in mapping:
        yy = 0
        for jj in mapping:
          f[ii, 0] += elem_fs[m, xx]
          q[ii, 0] += elem_qs[m, xx]
          j[ii, 0] += elem_js[m, xx]
          yy += 1
        xx += 1

      RHS = f + q + j
    return sps.csc_matrix(RHS)

  def assemble_bilinear_forms(self):
      for g in xrange(self.n_grp):
          self.assemble_group_bilinear_forms(g)

  def assemble_group_bilinear_forms(self,g):
    '''@brief Function used to assemble bilinear forms
    Must be called only once.
    '''
    A = sps.lil_matrix((nodes, nodes))
    B = sps.lil_matrix((nodes, nodes))
    C = sps.lil_matrix((nodes, nodes))

    elem_As = [] # list of elementary matrices based on material
    elem_Bs = []
    elem_Cs = []

    for m in num_materials:
      # TODO: Retrieve cross-sections for cell
      sigt = self.mat.sigt[m, g] # ask josh
      D = self.mat.D[m, g]
      siga = self.mat.siga[m, g]
      sigf = self.mat.sigf[g]
      nu = self.mat.nu[g]
        
      # TODO: Calculate kappa
      kappa = 1

      # TODO: Make function called basis_coefficients
      # Find Local Basis Functions
      # Set up Vandermonde Matrix
      V = np.array([[1, 0, 0, 0],
                  [1, 0, h, 0],
                  [1, h, 0, 0],
                  [1, h, h, h**2]])
      C = np.linalg.inv(V)
        
      # Assemble Elementary Matrix
      # A_ab = area(partial_ax*partial_bx + partial_ay*partial_by)
      area = h**2
      perimeter = 4*h
      elem_A = np.zeros((4, 4))
      elem_B = np.zeros((4, 4))
      elem_C = np.zeros((4, 4))

      for ii in xrange(4):
        for jj in xrange(4):
            partial_ax = C[1, ii]+C[3, ii]*V[ii, 2]
            partial_bx = C[1, jj]+C[3, jj]*V[jj, 2]
            partial_ay = C[2, ii]+C[3, ii]*V[ii, 1]
            partial_by = C[2, jj]+C[3, jj]*V[jj, 1]
            grad_ab = partial_ax*partial_bx + partial_ay*partial_by
            elem_A[ii,jj] = D*area*grad_ab
            elem_B[ii, jj] = driftVec*area*(partial_bx + partial_by)
            elem_D[ii,jj] = .5*kappa*perimeter
      
      # Append elementary matrix to list of matrices by material
      elem_As.append(elem_A)
      elem_Bs.append(elem_B)
      elem_Cs.append(elem_C)

    # Assemble Global Matrix
    for cell in xrange(n):
      cell_i = cell//int(np.sqrt(n))
      cell_j = cell%int(np.sqrt(n))

      m = # find material number of cell

      # Global to Local node mapping
      """Nodes are flattened by row with row x=0 going first.
      That means node x0,y0 in cell0 has a global index of 0
      and node x1,y1 in cell0 has a global index of nodes_x+1"""
      mapping = np.array([cell+cell_i, cell+cell_i+1, 
                            cell+cell_i+n_x+1, cell+cell_i+n_x+2])
      xx = 0
      for ii in mapping:
        yy = 0
        for jj in mapping:
          A[ii, jj] += elem_A[m, xx, yy]
          B[ii, jj] += elem_B[m, xx, yy]
          C[ii, jj] += elem_C[m, xx, yy]
          yy += 1
        xx += 1

    LHS = A + B + C
    return sps.csc_matrix(LHS)


  def assemble_group_linear_forms(self,g):
    '''@brief Function used to assemble linear forms for Group g
    '''
    assert 0<=g<=self.mat.n_group, 'Group index out of range'
    # TODO: fill in linear form assembly code
    # Scattering + fixed source 

  def solve_in_group(self):
    '''@brief Called to solve direction by direction inside Group g
    @param g Group index
    '''
    linear_forms = self.assemble_linear_forms()
    bilinear_form = self.assemble_bilinear_forms()

    for g in xrange(self.n_group):
      sps.linalg.spsolve(bilinear_form, linear_forms[g])



