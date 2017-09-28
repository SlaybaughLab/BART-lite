import numpy as np
import scipy.sparse as sps

class NDA(object):
  def __init__(self, mat):
    """ @brief constructor of NDA class

    @param mat of type material.material. """
    
    self.mat = mat

  def assemble_fixed_linear_forms(self):
    '''@brief a function used to assemble fixed source or fission source on the
    rhs for all components
    Generate numpy arrays and put them in self.
    '''
    f = sps.lil_matrix((nodes, 1))
    q = sps.lil_matrix((nodes, 1))
    j = sps.lil_matrix((nodes, 1))

    # TODO: Retrieve cross-sections for cell
    sigf = self.mat.sigf[g]
    nu = self.mat.nu[g]
        
    # TODO: Calculate previous flux
    flux_prev = 1 
        
        
    # TODO: Calculate j
    j = 1

    # Assemble LHS
    area = h**2
    perimeter = 4*h
    elem_f = np.zeros(4)
    elem_q = np.zeros(4)
    elem_j = np.zeros(4)

    for ii in xrange(4):
      elem_f[ii] = nu*sigf*flux_prev*area
      elem_q[ii] = q*area
      elem_j[ii] = 2*j*perimeter

    for cell in xrange(n):
      cell_i = cell//int(np.sqrt(n))
      cell_j = cell%int(np.sqrt(n))

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
          f[ii, 0] += elem_b[xx]
          q[ii, 0] += elem_b[xx]
          j[ii, 0] += elem_b[xx]
          yy += 1
        xx += 1

    RHS = f + q + j
    return sps.csc_matrix(RHS)

  def assemble_bilinear_forms(self):
    '''@brief Function used to assemble bilinear forms
    Must be called only once.
    '''
    A = sps.lil_matrix((nodes, nodes))
    B = sps.lil_matrix((nodes, nodes))
    C = sps.lil_matrix((nodes, nodes))

    # TODO: Retrieve cross-sections for cell
    sigt = self.mat.sigt[g] # ask josh
    D = 1/(3*sigt)
    siga = sigt - self.mat.sigs[g]
    sigf = self.mat.sigf[g]
    nu = self.mat.nu[g]
      
    # TODO: Calculate previous flux
    flux_prev = 1 
        
    # TODO: Calculate kappa
    kappa = 1
        
    # TODO: Calculate j
    j = 1

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
   
    # Assemble Global Matrix
    for cell in xrange(n):
      cell_i = cell//int(np.sqrt(n))
      cell_j = cell%int(np.sqrt(n))

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
          A[ii, jj] += elem_A[xx, yy]
          B[ii, jj] += elem_B[xx, yy]
          C[ii, jj] += elem_C[xx, yy]
          yy += 1
        xx += 1
    LHS = A + B + C
    return sps.csc_matrix(LHS)

  
  def assemble_linear_forms(self):
    '''@brief Function used to assemble all linear forms
    Different from assemble_group_linear_forms, which assembles linear forms
    for a specific group, this function calls assemble_group_linear_forms to
    assemble linear forms for all groups
    --------
    This function will only be called when NDA is used.
    '''
    return [self.assemble_group_linear_forms(g) 
              for g in xrange(self.mat.n_group)]


  def assemble_group_linear_forms(self,g):
    '''@brief Function used to assemble linear forms for Group g
    '''
    assert 0<=g<=self.mat.n_group, 'Group index out of range'
    # TODO: fill in linear form assembly code
  
  def solve_in_group(self):
    '''@brief Called to solve direction by direction inside Group g
    @param g Group index
    '''
    linear_forms = self.assemble_linear_forms()
    bilinear_form = self.assemble_bilinear_forms()

    for g in xrange(self.n_group):
      sps.linalg.spsolve(bilinear_form, linear_forms[g])



