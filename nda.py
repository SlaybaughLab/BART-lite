import numpy as np
from scipy import sparse as sps
from scipy.sparse import linalg as sla
from numpy.linalg import norm

from formulations import Formulation
from elem import Elem

class NDA(Formulation):
    def __init__(self, mat_cls, mesh_cls, prob_dict):
        Formulation.__init__(self, mat_cls, mesh_cls, prob_dict)
        # equation name
        self._name = 'nda'
        # problem type
        self._do_ua = prob_dict['do_ua']
        # total number of components: keep consistency with HO
        self._n_tot = self._n_grp
        # linear algebra objects
        self._sys_rhses = {k:np.ones(self._n_dof) for k in xrange(self._n_tot)}
        self._fixed_rhses = {k:np.zeros(self._n_dof) for k in xrange(self._n_tot)}
        # all material
        self._sigrs = mat_cls.get('sig_r')
        # derived material properties
        self._sigrs_ua = mat_cls.get('sig_r_ua')
        self._dcoefs_ua = mat_cls.get('diff_coef_ua')
        self._is_eigen = prob_dict['is_eigen_problem']
        # fission source
        self._global_fiss_src = self._calculate_fiss_src()
        self._global_fiss_src_prev = self._global_fiss_src


    def name(self):
        return self._name

    def assemble_bilinear_forms(self, ho_cls=None, correction=False):
        '''@brief A function used to assemble bilinear forms of NDA for current
        iterations

        @param correction A boolean used to determine if correction terms are
        assembled. By default, it's not. In this case, the bilinear form is typical
        diffusion
        '''
        # TODO: Boundary is assumed to be reflective so kappa will not be handled
        streaming,mass = self._elem.streaming(),self._elem.mass()
        if correction:
            assert ho_cls is not None, 'ho_cls has to be filled in for NDA correction'
        # basic diffusion Elementary matrices
        diff_mats = {}
        # Elementary correction matrices
        corx,cory,sigt,dcoef = self._elem.corx(),self._elem.cory(),0,0
        for g in xrange(self._n_grp):
            self._sys_mats[g] = sps.lil_matrix((self._mesh.n_node(),self._mesh.n_node()))
            for mid in self._mids:
                sigt,sigr,dcoef = self._sigts[mid][g],self._sigrs[mid][g],self._dcoefs[mid][g]
                diff_mats[(g,mid)] = (dcoef*streaming + sigr*mass)
        # preassembled matrices for upscattering acceleration
        if self._do_ua:
            self._sys_mats['ua'] = sps.lil_matrix((self._mesh.n_node(),self._mesh.n_node()))
            for mid in self._mids:
                dcoef_ua,sigr_ua = self._dcoefs_ua[mid],self._sigrs_ua[mid]
                # basic elementary diffusion matrices for upscattering acceleration
                diff_mats[('ua',mid)] = (dcoef_ua*streaming + sigr_ua*mass)

        # loop over cells for assembly
        for cell in self._mesh.cells():
            # get global dof index and mat id
            idx,mid = cell.get('id'),cell.global_idx()
            # corrections for all groups in current cell and ua
            corr_vecs = {}
            if correction:
                corr_vecs = ho_cls.calculate_nda_cell_correction(mat_id=mid,idx=idx)
            for g in xrange(self._n_grp):
                # if correction is asked
                cor_mat = np.zeros((4,4))
                if correction:
                    # calculate NDA correction in HO class
                    corr_vecs[g]=ho_cls.calculate_nda_cell_correction(
                    g=g, mat_id=mid, idx=idx)
                    # TODO: fixed the "9"
                    for i in xrange(9):
                        # x-component
                        cor_mat += corr_vecs['x_comp'][g][i]*corx[i]
                        # y-component
                        cor_mat += corr_vecs['x_comp'][g][i]*cory[i]
                    diff_mats[(g,mid)] += cor_mat
                # assemble global system
                for ci in xrange(4):
                    for cj in xrange(4):
                        self._sys_mats[g][idx[ci],idx[cj]]+=diff_mats[(g,mid)][ci,cj]

            # if we do upscattering acceleration
            if self._do_ua:
                # assemble global system of ua matrix without correction
                for ci in xrange(4):
                    for cj in xrange(4):
                        self._sys_mats['ua'][idx[ci],idx[cj]]+=diff_mats[('ua',mid)][ci,cj]

                # correction matrix for upscattering acceleration
                cor_mat_ua = np.zeros((4,4))
                if correction:
                    for i in xrange(len(corr_vecs[0])):
                        cor_mat_ua+=(corr_vecs['x_ua'][i]*corx[i]+
                                     corr_vecs['y_ua'][i]*cory[i])
                        diff_mats[('ua',mid)] += cor_mat_ua[ci,cj]
                # mapping UA matrix to global
                for ci in xrange(4):
                    for cj in xrange(4):
                        self._sys_mats['ua'][idx[ci],idx[cj]]+=diff_mats[('ua',mid)][ci,cj]

        # Transform system matrices to CSC format
        for g in xrange(self._n_grp):
            self._sys_mats[g] = sps.csc_matrix(self._sys_mats[g])
        if self._do_ua:
            self._sys_mats['ua'] = sps.csc_matrix(self._sys_mats['ua'])

    def assemble_fixed_linear_forms(self, sflxes_prev=None):
        '''@brief  function used to assemble linear form for fixed source or fission
        source
        '''
        # scale the fission xsec by keff
        scaled_fiss_xsec = {k:v/self._keff for k,v in self._fiss_xsecs}
        for g in xrange(self._n_grp):
            fixed_rhses[g] = np.array(self._mesh.n_node())
            for cell in self._mesh.cells():
                idx,mid,fiss_src = cell.global_idx(),cell.get('id'),np.zeros(4)
                for gi in filter(lambda: scaled_fiss_xsec[mid][g,x]>1.0e-14,xrange(self._n_grp)):
                    sflx_vtx = self._sflxes[g][idx] if not sflxes_prev else \
                               sflxes_prev[g][idx]
                    fiss_src += scaled_fiss_xsec[mid][g,gi]*np.dot(mass,sflx_vtx)
                self._fixed_rhses[g][idx] += fiss_src

    def _assemble_group_linear_forms(self, g):
        '''@brief A function used to assemble linear form for upscattering acceleration
        '''
        # NOTE: due to pass-by-reference feature in Python, we have to make
        # deep copy of fixed rhs instead of using "="
        np.copyto(self._sys_rhses[g], self._fixed_rhses[g])
        # get mass matrix
        mass = self._elem.mass()
        for cell in mesh.cells:
            idx,mid = cell.global_idx(),cell.get('id')
            sigs = self._sigses[mid][g,:]
            scat_src = np.zeros(4)
            for gi in filter(lambda x: sigs[x]>1.0e-14 and x!=g,xrange(self._n_grp)):
                sflx_vtx = self._sflxes[gi][idx]
                scat_src += sigs[gi] * np.dot(mass, sflx_vtx)
            self._sys_rhses[g][idx] += scat_src

    def _assemble_ua_linear_form(self, sflxes_old):
        '''@brief A function used to assemble linear form for upscattering acceleration
        '''
        assert len(sflxes_old)==self._n_grp, \
        'old scalar fluxes should have the same number of groups as current scalar fluxes'
        mass = self._elem.mass()
        self._sys_rhses['ua'] *= 0.0
        for cell in self._mesh.cells():
            idx,mid,scat_src_ua = cell.global_idx(),cell.get('id'),np.zeros(4)
            for g in xrange(self._g_thr,self._n_grp-1):
                for gi in xrange(g+1,self._n_grp):
                    sigs = self._sigses[mid][g,gi]
                    if sigs>1.0e-14:
                        dsflx_vtx = self._sflxes[gi][idx]-sflxes_old[g][idx]
                        scat_src_ua += sigs*np.dot(mass,dsflx_vtx)
            self._sys_rhses['ua'][idx] += scat_src_ua

    def solve_in_group(self,g):
        assert 0<=g<self._n_grp, 'Group index out of range'
        self._assemble_group_linear_forms(g)
        if g not in self._lu:
            # factorize it if not yet
            self._lu[g] = sla.splu(self._sys_mats[g])
        # direct solve
        self._sflxes[g] = self._lu[g].solve(self._sys_rhses[g])

    #NOTE: this function has to be removed if abstract class is implemented
    def calculate_keff(self):
        assert not self._is_eigen, 'only be called in eigenvalue problems'
        # update the previous fission source and previous keff
        self._global_fiss_src_prev,self._keff_prev = self._global_fiss_src,self._keff
        # calculate the new fission source
        self._calculate_fiss_src()
        # calculate the new keff
        self._keff = self._keff_prev * self._global_fiss_src / self._global_fiss_src_prev
        return self._keff

    #NOTE: this function has to be removed if abstract class is implemented
    def calculate_sflx_diff(self, sflxes_old, g):
        '''@brief function used to generate ho scalar flux for Group g using
        angular fluxes

        @param sflx_old Scalar flux from previous generation
        @param g The group index
        @return double The relative difference between new and old scalar flux
        '''
        # return the l1 norm relative difference
        return norm((self._sflxes[g]-sflxes_old[g]),1) / norm(self._sflxes[g],1)

    #NOTE: this function has to be removed if abstract class is implemented
    def update_sflxes(self, sflxes_old, g):
        '''@brief A function used to update scalar flux for group g

        @param sflxes_old A dictionary
        @param g Group index
        '''
        np.copyto(sflxes_old[g], self._sflxes[g])

    def update_ua(self):
        '''@brief A function used to update the scalar fluxes after upscattering
        acceleration
        '''
        for g in xrange(self.g_thr,self._n_grp):
            self._sflxes[g] += self._sflxes['ua']

    def clear_factorization(self):
        '''@brief A function used to clear all the factorizations after NDA is dictionaries
        for current iteration

        Every outer iterations, NDA equations are modified so previous factorizations
        are no longer suitable and must be cleared and redone.
        '''
        self._lu.clear()

    def solve_ua(self):
        # assemble ua
        self._assemble_ua_linear_form()
        if 'ua' not in self._lu:
            # factorize it if not yet
            self._lu['ua'] = sla.splu(self._sys_mats['ua'])
        # direct solve
        self._sflxes['ua'] = self._lu['ua'].solve(self._sys_rhses['ua'])

    def get_sflxes(self, g):
        '''@brief Function called outside to retrieve the scalar flux value for Group g

        @param g Target group number
        '''
        return self._sflxes[g]

    def get_sflx_vtx(self, g, idx):
        return self._sflxes[g][idx]

    def get_keff(self):
        '''@brief A function used to retrieve keff

        @return keff calculated in SAAF class
        '''
        return self._keff

    def do_ua(self):
        return self._do_ua

    def n_dof(self):
        return self._mesh.n_node()

    def n_grp(self):
        return self._n_grp

    def g_thr(self):
        return self._g_thr
