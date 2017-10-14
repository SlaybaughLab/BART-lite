import numpy as np
from mg_iterations import MG
'''
class used to perform eigenvalue calculations
'''
class Eigen(object):
    def __init__(self):
        # basic data
        self._keff = 1.0
        self._keff_prev = 1.0

        self._tol = 1e-5
        self._k_tol = 1e-5
        # multigroup class
        self._mg = MG()

    def do_iterations(self, ho_cls, nda_cls=None):
        '''@brief Function to be called outside for eigenvalue problems

        @param ho_cls The HO class instance
        @param nda_cls NDA instance
        '''
        n_dof,n_grp = ho_cls.n_dof(),ho_cls.n_grp()
        if not nda_cls:
            # assemble bilinear forms
            ho_cls.assemble_bilinear_forms()
            # NDA is not used in this calse
            self.eigen_iterations(ho_cls)
        else:
            assert nda_cls is not None, 'NDA class has to be provided for HOLO calculations'
            # initialize scalar fluxes from previous finished NDA solve
            sflxes_eig_prev_nda = {g:np.ones(n_dof) for g in xrange(n_grp)}
            # we assemble and solve with diffusion first (namely, no correction)
            nda_cls.assemble_bilinear_forms(nda_cls=None, correction=False)
            self.eigen_iterations(nda_cls)
            e_k,e_p = 1.,1.
            # get k
            keff_nda = nda_cls.get_keff()
            # assemble ho bilinear form
            ho_cls.assemble_bilinear_forms()
            while e_k>self._k_tol or e_p>self._tol:
                # update keff nda
                keff_prev_nda = keff_nda
                # update sflx from previous NDA iteration
                for g in xrange(n_grp):
                    nda_cls.update_sflxes(sflxes_eig_prev_nda,g)
                # solve once per component in HO
                ho_cls.solve_all_groups(nda_cls)
                # assemble NDA bilinear forms with HO correction
                nda_cls.assemble_bilinear_forms(ho_cls=ho_cls,correction=True)
                # assemble nda bilinear forms
                self.eigen_iterations(nda_cls)
                # get keff from NDA
                keff_nda = nda_cls.get_keff()
                # calculate error for k and phi in NDA
                e_k = abs((keff_nda-keff_prev_nda)/keff_nda)
                e_p = max(nda_cls.calculate_sflx_diff(sflxes_eig_prev_nda,g)
                          for g in xrange(n_grp))

    def eigen_iterations(self, equ_cls):
        '''@brief Function to be called in do_iterations
        '''
        n_dof,n_grp = equ_cls.n_dof(),equ_cls.n_grp()
        # initialize scalar fluxes from previous eigen iteration
        sflxes_eig_prev = {g:np.ones(n_dof) for g in xrange(n_grp)}
        ep,ek,keff, = 1.0,1.0,1.0
        while ep>self._tol and ek>self._k_tol:
            # update scalar flux from previous iteration
            for g in xrange(n_grp):
                equ_cls.update_sflxes(sflxes_eig_prev,g)
            # assemble for the fission source
            equ_cls.assemble_fixed_linear_forms(sflxes_prev=sflxes_eig_prev)
            # perform multigroup iteration to convergence
            self._mg.mg_iterations(equ_cls=equ_cls)
            # update keff
            keff_prev,keff = keff,equ_cls.calculate_keff()
            # calculate keff error
            ek = abs((keff-keff_prev)/keff)
            # calculate error of scalar flux in eigen iterations
            ep = max(equ_cls.calculate_sflx_diff(g) for g in xrange(self._n_grp))
