from nose.tools import *
from material import material
from material import _mat
import numpy as np

class TestFunctionality:

    @classmethod
    def setup_class(cls):
        filename = './tests/testData/test_mat.xml'
        NSfile = './tests/testData/test_mat_nonsource.xml'
        cls.testmat = _mat(filename, grps = 2)
        cls.testNSmat = _mat(NSfile, grps = 2)

    def test_mat_gen_prop(self):
        """ Reading a file should save the correct values to class vars. """
        ok_(type(self.testmat.prop["nu"]) == float, "Nu should be a float")
        ok_(self.testmat.prop["nu"] == 2.3, "Nu should be the correct value")
        ok_(self.testmat.gen['id'] == 'test_mat', "Mat ID should have correct value")

    def test_mat_xsec(self):
        """ Reading a file should save the correct xsec values """
        ok_(type(self.testmat.xsec['sig_t']) == np.ndarray, "Cross-section should be a numpy array")
        ok_(np.all(self.testmat.xsec['sig_t'] == [20.0, 30.0]), "Cross-sections should have correct values")


    def test_mat_gconst(self):
        """ Reading a file should save the correct gconst values """
        ok_(type(self.testmat.gconst['chi']) == np.ndarray, "Group constant should be a numpy array")
        ok_(np.all(self.testmat.gconst['chi'] == [0.5, 1.0]), "Group constant should have the correct value")

    def test_mat_issource(self):
        ok_(self.testmat.isSource, "Fission material be marked as a source")
        ok_(not self.testNSmat.isSource, "Non source material should be marked as a non-source")

       
class TestErrors:

    @classmethod
    def setup_class(cls):
        filename = './tests/testData/test_mat.xml'
        NSfile = './tests/testData/test_mat_nonsource.xml'
        cls.testmat = _mat(filename, grps = 2)
        cls.testNSmat = _mat(NSfile, grps = 2)

    @raises(KeyError)
    def test_mat_bad_structure(self):
        """ Specifying a group structure not in the material file should return
        a Key error """
        filename = './tests/testData/test_mat.xml'
        badMat = _mat(filename, grps = 3)

    @raises(AssertionError)
    def test_mat_bad_filename(self):
        """ Reading a bad filename should return an assertion error. """
        filename = './tests/testData/badname.xml'
        badMat = _mat(filename, grps = 2)

    @raises(RuntimeError)
    def test_no_id(self):
        """ Uploading a material with no id should return a runtime error. """
        filename = './tests/testData/test_no_id.xml'
        badMat = _mat(filename, grps = 2)
        
    @raises(RuntimeError)
    def test_xsec_sizes(self):
        """ All cross section arrays should be the same size. """
        filename = './tests/testData/test_bad_xsec.xml'
        badMat = _mat(filename, grps = 2)

class TestWarnings:

    @classmethod
    def setup_class(cls):
        filename = './tests/testData/test_mat.xml'
        NSfile = './tests/testData/test_mat_nonsource.xml'
        cls.testmat = _mat(filename, grps = 2)

    
