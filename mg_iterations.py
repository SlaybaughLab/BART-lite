import numpy as np

'''
class used to perform multigroup calculations
'''
class MG(object):
    def __init__(self):
        # TODO: address constructor
        self.n_grp
        self.g_thr # the first group we see upscattering
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
            ho_cls.assemble_fixed_linear_forms()
            self.mg_iterations(ho_cls)
        else:
            # TODO: add NDA
            raise NotImplementedError

    def mg_iterations(self, equ_cls):
        '''@brief Function used to do multigroup iterations

        @param equ_cls Equation class instance. Could be NDA or HO depending on
        problem definition.
        Only to be called by self.do_iterations or in eigenvalue iterations
        '''
        # Solve for fast and epithermal groups
        for g in xrange(0,self.g_thr):
            equ_cls.assemble_linear_forms(g)
            equ_cls.solve_in_group(g)
            equ_cls.generate_sflx(g)
        # Solve for thermal groups
        e,sflxes_old_mg = 1.0,[0]*(self.n_grp-self.g_thr)
        while e>self._tol:
            e = self._tol * 0.1
            for g in xrange(self.g_thr, self.n_grp):
                equ_cls.assemble_linear_forms(g)
                equ_cls.solve_in_group(g)
                # calculate mg iteration error for group g
                equ_cls.calculate_sflx_diff(sflxes_old_mg,g)
                # get the max iteration error from all thermal groups
                e = max(e, equ_cls.get_sflx_diff, g)
