from nose.tools import *
from bart_io import mat_io

class TestClass:

    @raises(AssertionError)
    def test_mat_io_nofile(self):
        """ mat_io function returns error if file is no found """
        mat_io(filename="testdata/badfile.xml")
