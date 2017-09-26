from nose.tools import *
from material import material
from material import _mat
import numpy as np

class TestClass:

    @classmethod
    def setup_class(cls):
        filename = './tests/testData/test_mat.xml'
        NSfile = './tests/testData/test_mat_nonsource.xml'
        cls.testmat = _mat(filename, grps = 2)
        cls.testNSmat = _mat(NSfile, grps = 2)

    @raises(AssertionError)
    def test_mat_bad_filename(self):
        """ Reading a bad filename should return an assertion error. """
        filename = './tests/testData/badname.xml'
        badMat = _mat(filename, grps = 2)

    def test_mat_read(self):
        """ Reading a file should save the correct values to class vars. """
        ok_(type(self.testmat.prop["nu"]) == float, "Nu should be a float")
        ok_(self.testmat.prop["nu"] == 2.3, "Nu should be the correct value")
        ok_(type(self.testmat.xsec['sig_t']) == np.ndarray, "Cross-section should be a numpy array")
        ok_(np.all(self.testmat.xsec['sig_t'] == [20.0, 30.0]), "Cross-sections should have correct values")

    def test_mat_issource(self):
        ok_(self.testmat.isSource, "Fission material be marked as a source")
        ok_(not self.testNSmat.isSource, "Non source material should be marked as a non-source")
        
    @raises(KeyError)
    def test_mat_bad_structure(self):
        """ Specifying a group structure not in the material file should return
        a Key error """
        filename = './tests/testData/test_mat.xml'
        badMat = _mat(filename, grps = 3)
