import numpy as np
'''
class used to perform multigroup calculations
'''


class MG(object):
    def __init__(self):
        # iteration tol
        self._tol = 1e-5

    def do_iterations(self, ho_cls, nda_cls=None):
        '''@brief Function to be called in fixed source problems

        @param ho_cls The HO class instance
        @param nda_cls NDA instance
        It is not supposed to be called in eigen value problems
        '''
        if not nda_cls:
            # NDA is not used in this calse
            # assemble matrices
            ho_cls.assemble_bilinear_forms()
            # assemble fixed source
            ho_cls.assemble_fixed_linear_forms()
            # multigroup calculations
            self.mg_iterations(ho_cls)
        else:
            # TODO: add NDA for fixed source problem without fission source
            raise NotImplementedError

    def mg_iterations(self, equ_cls):
        '''@brief Function used to do multigroup iterations

        @param equ_cls Equation class instance. Could be NDA or HO depending on
        problem definition.
        Only to be called by self._do_iterations or in eigenvalue iterations
        '''
        # get number of groups and first thermal group
        n_dof, n_grp, g_thr = equ_cls.n_dof(), equ_cls.n_grp(), equ_cls.g_thr()
        # sflxes from previous MG iteration
        sflxes_mg_prev = {g: np.ones(n_dof) for g in xrange(g_thr, n_grp)}
        # Solve for fast and epithermal groups
        for g in xrange(0, g_thr):
            equ_cls.solve_in_group(g)
        # Solve for thermal groups
        e = 1.0
        while e > self._tol:
            for g in xrange(g_thr, n_grp):
                # update old mg flux
                equ_cls.update_sflxes(sflxes_mg_prev, g)
                # assemble linear form and solve in group
                equ_cls.solve_in_group(g)
                # calculate mg iteration error for group g
            if equ_cls.name() == 'nda' and equ_cls.do_ua():
                # solve ua equation
                equ_cls.solve_ua()
                # update nda sflx after upscattering acceleration
                equ_cls.update_ua()
            # calculate iteration errors in multigroup iterations
            e = max(
                equ_cls.calculate_sflx_diff(sflxes_mg_prev, g)
                for g in xrange(g_thr, n_grp))
