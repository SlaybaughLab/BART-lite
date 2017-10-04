import numpy as np
import scipy.sparse as sps

class NDA(object):
    def __init__(self, mat, mat_map, mesh):
        """ @brief constructor of NDA class

        @param mat of type material.material. """
        # Geometry Information
        self.n = mesh.CELLS # number of cells
        self.h = mesh.CELL_LENGTH # length of cells
        self.n_x = int(domain_length/self.h) # number of mesh cells in each direction
        self.nodes_x = int(self.n_x + 1) # number of node points in each direction
        self.nodes = int(self.nodes_x**2) # number of total node points
        # Materials Information
        self.mat = mat # materials library
        self.mat_map = mat_map # materials map
        self.num_materials = len(mat.ids()) # extract number of materials 
        self.n_grp = mat.n_grps # number of groups
        # Matrices and Vectors
        self.sys_mats = [0] * self.n_grp
        self.sys_rhses = [0] * self.n_grp
        self.fixed_rhses = [0] * self.n_grp

    def drift_vector(self, sflxes_prev=None, g):
        weight = .5 # gauss-legendre weight for 2 points in half-interval
        driftVec = np.array(n)
        for cell in xrange(n):
            cell_i = cell//int(np.sqrt(n))
            cell_j = cell%int(np.sqrt(n))

            mat_id = mat_map.get('id', n) # retrieve material id
            m = materials.index(mat_id)
            inv_sigt = self.mat.get('inv_sigt', mat_id=m)[g]
            # retrieve fluxes at local nodes 0-3
            mapping = np.array([cell+cell_i, cell+cell_i+1, 
                        cell+cell_i+n_x+1, cell+cell_i+n_x+2])
            lflx = sflxes_prev[mapping] #local fluxes

            # calculate fluxes at gauss-legandre points
            glflx = np.zeros(4)
            glpoint = .5*np.sqrt(1/3.0) + .5 # g-l point for 2 points on half interval
            deltax0 = np.abs(lflx[1] - lflx[0])
            deltay0 = np.abs(lflx[0] - lflx[2])
            deltax3 = np.abs(lflx[2] - lflx[3])
            deltay3 = np.abs(lflx[1] - lflx[3])
            glflx[0] = lflx[0] + (deltax0 + deltay0)*glpoint
            glflx[1] = lflx[1] + (deltay3 - deltax0)*glpoint
            glflx[2] = lflx[2] + (deltax3 - deltay0)*glpoint
            glflx[3] = lflx[3] - (deltax3 + deltay3)*glpoint

            # TODO: Actual calculation for drift vector
            for m in xrange(4):
                num += weight*(inv_sigt) 
                den += weight*glflx[m]
            driftVec[cell] = num/den
        return driftVec
            
    def assemble_fixed_linear_forms(self, sflxes_prev=None):
        '''@brief a function used to assemble fixed source or fission source on the
        rhs for all components
        Generate numpy arrays and put them in self.
        '''
        f = sps.lil_matrix((nodes, 1)) # just use numpy array
        q = sps.lil_matrix((nodes, 1))
        j = sps.lil_matrix((nodes, 1))

        elem_fs = [] # list of elementary vector based on material
        elem_qs = []
        elem_js = []
    
        area = h**2 # retrieve h
        perimeter = 4*h
        materials = self.mat.ids()

        # Set up elementary matrices for each material and group 
        for g in xrange(n_grp):
            for m in materials:
            
                # Retrieve cross-sections for cell
                sigf = self.mat.get('sig_f', mat_id=m)[g] # material and group 
                nu = self.mat.get('nu', mad_id=m)
        
                # TODO: Calculate j
                j = 1

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

                mat_id = mat_map.get('id', n) # retrieve material id
                m = materials.index(mat_id)
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

            fixed_rhs = f + q + j
            fixed_rhs = sps.csc_matrix(fixed_rhs)
            fixed_rhses[g] = fixed_rhs

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

        area = h**2
        perimeter = 4*h
        materials = self.mat.ids()

        # Find Local Basis Functions
        # Set up Vandermonde Matrix
        # TODO: use one in elem.py
        V = np.array([[1, 0, 0, 0],
                  [1, 0, h, 0],
                  [1, h, 0, 0],
                  [1, h, h, h**2]])
        C = np.linalg.inv(V)

        for m in materials:
            sigt = self.mat.get('sig_t', mat_id=m)[g]
            D = self.mat.get('D', mat_id=m)[g]
            siga = self.mat.get('sig_a', mat_id=m)[g]
            sigf = self.mat.get('sig_f', mat_id=m)[g]
            nu = self.mat.get('nu', mat_id=m)
        
            # TODO: Calculate kappa
            kappa = 1
        
            # Assemble Elementary Matrix
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

            mat_id = mat_map.get('id', n) # retrieve material id
            m = materials.index(mat_id)   # find material index

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

        lhs = A + B + C
        return sps.csc_matrix(lhs) # don't return

    def assemble_group_linear_forms(self,g):
        '''@brief Function used to assemble linear forms for Group g
        '''
        assert 0<=g<=self.mat.n_group, 'Group index out of range'
        # TODO: fill in linear form assembly code
        # Scattering + fixed source 

    def assemble_bilinear_forms(self):
        for g in xrange(self.n_grp):
            self.sys_mats[g] = self.assemble_group_bilinear_forms(g)

    def solve_in_group(self, g):
        '''@brief Called to solve direction by direction inside Group g
        @param g Group index
        '''
        assert 0<=g<self.n_grp, 'Group index out of range'
        # if not factorized, factorize the the HO matrices
        if self.lu[g]==0:
            self.lu[g] = sla.splu(self.sys_mats[g])
        self.aflxes[g] = self.lu[g].solve(self.sys_rhses[g])



