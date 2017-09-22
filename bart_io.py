import xml.etree.ElementTree as ET
import os, sys

def mat_io(filename):
    assert os.path.exists(filename), "File does not exist"
