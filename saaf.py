# Python Standard Imports
from itertools import product
# Third Party Tools
import numpy as np
from scipy import sparse as sps
from scipy.sparse import linalg as sla
from numpy.linalg import norm

# Files from This Code
from elem import Elem
from aq import AQ
from formulations import Formulation


class SAAF(Formulation):
    def __init__(self, mat_cls, mesh_cls, prob_dict):
        Formulation.__init__(self, mat_cls, mesh_cls, prob_dict)
        # name of the Equation
        self._name = 'saaf'
        # material data
        self._isigts = mat_cls.get('inv_sig_t')
        # derived material data
        self._ksi_ua = mat_cls.get('ksi_ua')
        # aq data in forms of dictionary
        self._aq = AQ(prob_dict['sn_order']).get_aq_data()
        self._n_dir = self._aq['n_dir']
        # total number of components in HO
        self._n_tot = self._n_grp * self._n_dir
        # linear algebra objects
        self._sys_rhses = {
            k: np.ones(self._n_dof)
            for k in xrange(self._n_tot)
        }
        self._fixed_rhses = {
            k: np.zeros(self._n_dof)
            for k in xrange(self._n_tot)
        }
        # mappings
        # TODO: change to bidirectional map
        self._comp = {gd: ct for ct, gd in enumerate(self._group_dir_pairs())}
        self._comp_grp = {ct: g for ct, (g,_) in enumerate(self._group_dir_pairs())}
        self._comp_dir = {ct: d for ct, (_,d) in enumerate(self._group_dir_pairs())}
        # local vectors
        self._rhs_mats = self._preassembly_rhs()
        self._aflxes = {k: np.ones(self._n_dof) for k in xrange(self._n_tot)}
        # previous scalar flux for ingroup calculations
        self._sflx_ig_prev = np.ones(self._n_dof)
        # source iteration tol
        self._tol = 1.0e-7
        self._is_eigen = True  # TODO: get from prob_dict


    def _group_dir_pairs(self):
        return product(xrange(self._n_grp), xrange(self._n_dir))

    def _group_dir_rhs(self, g, d, isigts, sigts):
        '''RHS for each group and direction'''
        ox, oy = self._aq['omega'][d]
        # streaming & mass part of rhs
        rhs_mat = (ox * self._elem.dxvu() + oy * self._elem.dyvu()
                   ) * isigts[g] + sigts[g] * self._elem.mass()
        return rhs_mat

    def _material_rhs(self, mid):
        '''RHS for each material'''
        sigts, isigts = self._sigts[mid], self._isigts[mid]
        return {(g, d): self._group_dir_rhs(g, d, isigts, sigts)
                for g, d in self._group_dir_pairs()}

    def _preassembly_rhs(self):
        return {mid: self._material_rhs(mid) for mid in self._mids}

    def _lhs_per_material(self, mid, g, d):
        # get omega_i * omega_j combinations
        oxox, oxoy, oyoy = self._aq['dir_prods'][d].values()
        sigt, isigt = self._sigts[mid][g], self._isigts[mid][g]
        # streaming lhs
        matx = isigt * (oxox * self._elem.dxdx() + oxoy *
                        (self._elem.dxdy() + self._elem.dydx()) +
                        oyoy * self._elem.dydy())
        # collision matrix
        matx += sigt * self._elem.mass()
        return matx

    def assemble_bilinear_forms(self):
        '''@brief Function used to assemble bilinear forms

        @param correction Boolean to determine if correction is needed. Only useful
        in NDA class
        '''
        # retrieve all the material properties
        for g, d in self._group_dir_pairs():
            # dict containing lhs local matrices for all materials for component i
            lhs_mats = {mid: self._lhs_per_material(mid, g, d) for mid in self._mids}
            # sys_mat: temp variable for system matrix for one component
            sys_mat = sps.lil_matrix((self._mesh.n_node(),
                                      self._mesh.n_node()))
            for cell in self._mesh.cells():
                # retrieving global indices and material id per cell
                idx, mid = cell.global_idx(), cell.get('id')
                # mapping local matrices to global
                for ci, cj in _elements_of_elem_matrix():
                    sys_mat[idx[ci], idx[cj]] += lhs_mats[mid][ci][cj]
                # boundary part
                if not cell.bounds():
                    continue
                # loop over
                for bd in cell.bounds().keys():
                    if self._aq['bd_angle'][(bd, d)] <= 0:
                        continue
                    #outgoing boundary assembly: retrieving omega*n and boundary mass matrices
                    odn, bd_mass = self._aq['bd_angle'][(bd, d)], self._elem.bdmt()[bd]
                    # mapping local vertices to global
                    for ci, cj in _elements_of_elem_matrix():
                        if bd_mass[ci][cj] > 1.0e-14:
                            sys_mat[idx[ci], idx[cj]] += odn * bd_mass[ci][cj]
            '''
            # use this section only for generating sparsity pattern
            if i==0:
                import matplotlib.pyplot as plt
                plt.spy(sys_mat, precision=0.1, markersize=.5)
                plt.show()
            '''
            # transform lil_matrix to csc_matrix for efficient computation
            self._sys_mats[g, d] = sps.csc_matrix(sys_mat)
        return self._sys_mats

    def assemble_fixed_linear_forms(self, sflxes_prev=None, nda_cls=None):
        '''@brief a function used to assemble fixed source or fission source on the
        rhs for all components

        Generate fission source. If nda_cls is not None, sflxes_prev is ignored
        '''
        if not nda_cls:
            assert sflxes_prev is not None, 'scalar flux must be provided'
        # get properties per str scaled by keff
        for cp in xrange(self._n_tot):
            # re-init fixed rhs. This must be done at the beginning of calling this function
            self._fixed_rhses[cp] = np.zeros(self._n_dof)
            # get group and direction indices
            g, d = self._comp_grp[cp], self._comp_dir[cp]
            for cell in self._mesh.cells():
                idx, mid = cell.global_idx(), cell.get('id')
                fiss_src, fiss_xsec = np.zeros(4), self._fiss_xsecs[mid][g]
                # get fission source contribution from ingroups
                for gi in filter(lambda j: fiss_xsec[j] > 1.0e-14,
                                 xrange(self._n_grp)):
                    sflx_vtx = sflxes_prev[gi][idx] if not nda_cls else \
                               nda_cls.get_sflx_vtx(gi, idx)
                    fiss_src += fiss_xsec[gi] * np.dot(self._rhs_mats[mid][(
                        g, d)], sflx_vtx)
                self._fixed_rhses[cp][idx] += fiss_src

    def _assemble_group_linear_forms(self, g, nda_cls=None):
        '''@brief Function used to assemble linear forms for Group g

        if NDA is used, nda_cls must be filled in. This function will not be called
        from outside. It will rather be called in solve_in_group or solve_all_groups
        '''
        assert 0 <= g < self._n_grp, 'Group index out of range'
        if nda_cls:
            assert nda_cls.name(
            ) == 'nda', 'Correct NDA class must be filled in'
        for d in xrange(self._n_dir):
            cp = self._comp[(g, d)]
            # get fixed/fission source
            # NOTE: due to pass-by-reference feature in Python, we have to make
            # deep copy of fixed rhs instead of using "="
            np.copyto(self._sys_rhses[cp], self._fixed_rhses[cp])
            # go through all cells
            for cell in self._mesh.cells():
                # get global dof indices and material ids
                idx, mid = cell.global_idx(), cell.get('id')
                # get scattering matrix for current cell
                sigs = self._sigses[mid]
                # calculate local scattering source
                scat_bd_src = np.zeros(4)
                for gi in filter(lambda x: sigs[g][x] > 1.0e-14,
                                 xrange(self._n_grp)):
                    # retrieve scalar flux at vertices
                    sflx_vtx = self._sflxes[gi][idx] if not nda_cls \
                    else nda_cls.get_sflx_vtx(gi, idx)
                    # calculate scattering source
                    scat_bd_src += sigs[g, gi] * np.dot(
                        self._rhs_mats[mid][(g, d)], sflx_vtx)
                # if it's boundary
                if cell.bounds():
                    for bd, tp in cell.bounds().items():
                        # incident boundary with reflective setting
                        if tp == 'refl' and self._aq['bd_angle'][(bd,
                                                                  d)] < 0.0:
                            r_dir = self._aq['refl_dir'][(bd, d)]
                            odn = abs(self._aq['bd_angle'][(bd, d)])
                            bd_mass = self._elem.bdmt()[bd]
                            bd_aflx = self._aflxes[self._comp[(g, r_dir)]][idx]
                            scat_bd_src += odn * np.dot(bd_mass, bd_aflx)
                self._sys_rhses[cp][idx] += scat_bd_src

    def _assemble_linear_forms(self, nda_cls):
        '''@brief A function call to assemble linear forms for all components once

        This function is to be called along with NDA providing keff and fluxes
        '''
        assert nda_cls is not None and nda_cls.name(
        ) == 'nda', 'NDA has to be passed in to call'
        # NOTE: this is not the most efficient way as there is no need to separating
        # the assembly process here
        self.assemble_fixed_linear_forms(sflxes_prev=None, nda_cls=nda_cls)
        for g in xrange(self._n_grp):
            self._assemble_group_linear_forms(g=g, nda_cls=nda_cls)

    def solve_all_groups(self, nda_cls):
        '''@brief A function call to solve all components once

        This function is to be called along with NDA providing rhs
        '''
        self._assemble_linear_forms(nda_cls=nda_cls)
        for g, d in self._group_dir_pairs():
            # TODO: remove i
            i = self._comp[(g, d)]
            if i not in self._lu:
                # factorization
                self._lu[i] = sla.splu(self._sys_mats[g, d])
            # direct solve for angular fluxes
            self._aflxes[i] = self._lu[i].solve(self._sys_rhses[i])

    def solve_in_group(self, g):
        '''@brief Called to solve direction by direction inside Group g

        @param g Group index
        '''
        assert 0 <= g < self._n_grp, 'Group index out of range'
        # Source iteration
        e = 1.0
        while e > self._tol:
            print 'in group SAAF error: ', e
            # assemble group rhses
            self._assemble_group_linear_forms(g)
            # copy scalar flux
            np.copyto(self._sflx_ig_prev, self._sflxes[g])
            self._sflxes[g] *= 0.0
            for d in xrange(self._n_dir):
                # if not factorized, factorize the the HO matrices
                cp = self._comp[(g, d)]
                if cp not in self._lu:
                    print 'factorize SAAF for component ', cp
                    self._lu[cp] = sla.splu(self._sys_mats[(g, d)])
                # solve direction d
                self._aflxes[cp] = self._lu[cp].solve(self._sys_rhses[cp])
                self._sflxes[g] += self._aq['wt'][d] * self._aflxes[cp]
            # calculate difference for SI convergence
            e = norm(self._sflx_ig_prev - self._sflxes[g], 1) / norm(
                self._sflxes[g], 1)

    def calculate_nda_cell_correction(self, mat_id, idx, do_ua=False):
        # TODO: address the "9"
        corrs = {
            'x_comp': np.zeros((self._n_grp, 9)),
            'y_comp': np.zeros((self._n_grp, 9))
        }
        for g in xrange(self._n_grp):
            dcoef = self._dcoefs[mat_id][g]
            isgit = self._isigts[mat_id][g]
            # retrive grad_aflx for all directions at quadrature points
            grad_aflxes_qp = {
                d: self._elem.get_grad_at_qps(
                    self._aflxes[self._comp[(g, d)]][idx])
                for d in xrange(self._n_dir)
            }
            sflxes_qp = self._elem.get_sol_at_qps(self._sflxes[g][idx])
            grad_sflx_qp = self._elem.get_grad_at_qps(self._sflxes[g][idx])
            # calculate correction for x and y components
            corx, cory = np.zeros(len(sflxes_qp)), np.zeros(len(sflxes_qp))
            for i in xrange(len(sflxes_qp)):
                # transport current
                tc = np.zeros(2)
                for d in xrange(self._n_dir):
                    # NOTE: 'wt_tensor' is equal to w*OmegaOmega, a 2x2 matrix
                    tc += np.dot(self._aq['wt_tensor'][d],
                                 grad_aflxes_qp[d][i])
                # minus diffusion current
                mdc = dcoef * grad_sflx_qp[i]
                # corrections
                corx[i], cory[i] = (isgit * tc + mdc) / sflxes_qp[i]
            # wrap corrections to corrs
            corrs['x_comp'][g], corrs['y_comp'][g] = corx, cory
        # do upscattering acceleration
        if do_ua:
            corrs['x_ua'] = np.dot(self._ksi_ua[mat_id],
                                   corrs['x_comp'][self._g_thr:, ])
            corrs['y_ua'] = np.dot(self._ksi_ua[mat_id],
                                   corrs['y_comp'][self._g_thr:, ])
        return corrs

def _elements_of_elem_matrix():
    """iterator over elements of elementary matrix. 
    Note 4x4 only works for square elements & bilinar basis functions."""
    return product(xrange(4), xrange(4))
