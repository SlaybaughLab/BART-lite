from nose.tools import *
from material import mat_lib
import numpy as np

testData_loc = './tests/testData/materials/'

class TestFunctionality:
    # Tests to verify the material library class
    
    @classmethod
    def setup_class(cls):
        files = ['test_mat.xml', 'test_mat2.xml']
        filelocs = [testData_loc + f for f in files]
        cls.lib = mat_lib(n_grps = 2, files = filelocs)

    # TEST CONSTRUCTOR ###############################################

    def test_init_with_list(self):
        """ Initializing with a list should upload correct files """
        eq_(len(self.lib.mats), 2)
        eq_([m.gen['id'] for m in self.lib.mats],
            ['test_mat', 'test_mat2'])

    @raises(RuntimeError)
    def test_init_same_mat_ids(self):
        """ Adding two materials with different ids should throw a
        runtime error"""
        files = ['test_mat.xml', 'test_mat.xml']
        filelocs = [testData_loc + f for f in files]
        badLib = mat_lib(n_grps = 2, files = filelocs)

    # ACCESS FUNCTIONS ###############################################

    def test_ids(self):
        """ Ids function should return correct values """
        ok_(all([id in self.lib.ids()
                 for id in ['test_mat', 'test_mat2']]))
