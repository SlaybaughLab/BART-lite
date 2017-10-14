import numpy as np

from elem import Elem


class Formulation(object):
    def __init__(self, mat_cls, mesh_cls, prob_dict):
        self._name = 'Formulation'
        # mesh data
        self._mesh = mesh_cls
        self._cell_length = mesh_cls.cell_length()
        # quantities of interest
        self._keff = 1.0
        self._keff_prev = 1.0
        # preassembly-interpolation data
        self._elem = Elem(self._cell_length)
        # material ids and group info
        self._mids = mat_cls.ids()
        self._n_grp = mat_cls.get('n_grps')
        self._g_thr = int(min(mat_cls.get('g_thermal').values()))
        self._dcoefs = mat_cls.get('diff_coef')
        self._sigts = mat_cls.get('sig_t')
        self._sigses = mat_cls.get_per_str('sig_s')
        self._fiss_xsecs = mat_cls.get('chi_nu_sig_f')
        self._nu_sigfs = mat_cls.get('nu_sig_f')
        # related to global matrices and vectors
        self._n_dof = mesh_cls.n_node()
        # linear algebra objects
        self._sys_mats = {}
        # scalar flux for current calculation
        self._sflxes = {k: np.ones(self._n_dof) for k in xrange(self._n_grp)}
        # linear solver objects
        self._lu = {}
        # fission source
        self._global_fiss_src = self._calculate_fiss_src()
        self._global_fiss_src_prev = self._global_fiss_src

    def _calculate_fiss_src(self):
        # loop over cells and groups and calculate nu_sig_f*phi
        # NOTE: the following calculation is using mid-point rule for integration
        # It will suffice only for constant,RT1 and bilinear finite elements.
        global_fiss_src = 0
        for cell in self._mesh.cells():
            idx, mid = cell.global_idx(), cell.get('id')
            nusigf = self._nu_sigfs[mid]
            for g in filter(lambda x: nusigf[x] > 1.0e-14,
                            xrange(self._n_grp)):
                sflx_vtx = sum(self._sflxes[g][idx])
                global_fiss_src += nusigf[g] * sflx_vtx
        return global_fiss_src

    def calculate_keff(self):
        assert self._is_eigen, 'only be called in eigenvalue problems'
        # update the previous fission source and previous keff
        self._global_fiss_src_prev, self._keff_prev = self._global_fiss_src, self._keff
        # calculate the new fission source
        self._global_fiss_src = self._calculate_fiss_src()
        # calculate the new keff
        self._keff = self._keff_prev * self._global_fiss_src / self._global_fiss_src_prev
        return self._keff

    def calculate_sflx_diff(self, sflxes_old, g):
        '''@brief function used to generate ho scalar flux for Group g using
        angular fluxes

        @param sflx_old Scalar flux from previous generation
        @param g The group index
        @return double The relative difference between new and old scalar flux
        '''
        # return the l1 norm relative difference
        return np.linalg.norm(self._sflxes[g] - sflxes_old[g], 1) / np.linalg.norm(self._sflxes[g], 1)

    def update_sflxes(self, sflxes_old, g):
        '''@brief A function used to update scalar flux for group g

        @param sflxes_old A dictionary
        @param g Group index
        '''
        np.copyto(sflxes_old[g], self._sflxes[g])

    def name(self):
        return self._name

    def get_sflxes(self, g):
        '''@brief Function called outside to retrieve the scalar flux value for Group g

        @param g Target group number
        '''
        return self._sflxes[g]

    def get_keff(self):
        '''@brief A function used to retrieve keff

        @return keff calculated in SAAF class
        '''
        return self._keff

    def n_dof(self):
        return self._mesh.n_node()

    def n_grp(self):
        return self._n_grp

    def g_thr(self):
        return self._g_thr
