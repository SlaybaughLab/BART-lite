import numpy as np
import warnings
from math import pi
import os, sys
import xml.etree.cElementTree as ET

class _mat():
    def __init__(self, filename, grps):
        """ Constructor of a single material, reads from a provided
        filename and parses the required data from the xml file

        Material properties are as follows:
        self.gen:     any tags in <material>, i.e. name, id
        self.prop:    any tags in material/prop, i.e. nu
        self.gconst:  any non-xsec tags in grp_struct, i.e. chi
        self.xsec:    any tags in xsec, i.e. sig_t, sig_a, sig_f
        self.derived: derived quantities:
          - inv_sig_t: inverse of <sig_t> if present
          - diff_coef: diffusion coef, if <sig_t> present
          - chi_nu_sig_f: chi*nu*sig_f if <nu> and <sig_f> present
        """

        # Verify file exists
        assert os.path.exists(filename), "Material file: " + filename\
            + " does not exist"
        
        self.n_grps  = grps
        self.gen     = {}     # General properties
        self.prop    = {}     # Physical properties
        self.gconst  = {}     # Group non cross-section data
        self.xsec    = {}     # Group cross-section data
        self.derived = {}     # Derived quantities

        self.__parse_XML__(filename, grps)  # Parse input XML file
        self.__validate__(filename)         # Validate material data
        
        if 'sig_t' in self.xsec:
            self.__derive_sig_t__()         # Calc sig_t derv. prop

        if self.isSource:
            self.__derive_fiss__()          # Calc fission derv. prop

    # INITIALIZATION FUNCTIONS ========================================

    def __derive_sig_t__(self):
        # Calculate derived quantities based on sig_t

        # Find non-zero entries of Sig_t
        non_zero = self.xsec['sig_t']!=0
        # Inverse Sig_t
        inv_sig_t = np.power(self.xsec['sig_t'], -1,
                                 where=non_zero)
        # Diffusion Coeff
        diff_coef = np.power(3. * self.xsec['sig_t'], -1,
                                 where=non_zero)

        self.derived.update({'diff_coef': diff_coef})
        self.derived.update({'inv_sig_t': inv_sig_t})            
        
    def __derive_fiss__(self):
        # Calculate derived quantities based on fission properties
        
        chi_nu_sig_f = np.multiply(self.gconst['chi'],
                                   self.prop['nu']*self.xsec['sig_f'])
        self.derived.update({'chi_nu_sig_f': chi_nu_sig_f})

        
            
    
    def __parse_XML__(self, filename, grps):
        # Parse the XML file
        
        # These are the tags that identify the different sections of
        # the XML file. They are here to make it easy to change it in
        # the future.
        tag_prop           = "prop"            # Phys. Properties
        tag_grp_structures = "grp_structures"  # Group structures
        tag_xsec           = "xsec"            # Cross-sections
        
        # Get data and root
        root = ET.parse(filename).getroot()

        # Parse top level tags, get roots for physical properties and
        # group structures
        for el in root.findall('./material/'):
            if el.tag == tag_prop:
                prop_root = el
            elif el.tag == tag_grp_structures:
                grp_structs = el
            else:
                self.__dict_add__(self.gen, el)
        
        # Parse physical properties
        for el in prop_root:
            self.__dict_add__(self.prop, el)

        # Find correct group structure, or throw error
        try:
            grp_root = grp_structs.findall(".*[@n='" +
                                           str(self.n_grps) + "']")[0]
        except IndexError:
            raise KeyError(filename + ": group structure for n=" +
                             str(self.n_grps) + " not found")

        # Parse non cross-section constant data
        for el in grp_root:
            if el.tag == tag_xsec:
                xsec_root = el
            else:
                self.__dict_add__(self.gconst, el)
                
        # Parse cross-sections
        for el in xsec_root.findall('*'):
            self.__dict_add__(self.xsec, el)

        if 'nu' in self.prop and 'sig_f' in self.xsec:
            self.isSource = True
        else:
            self.isSource = False
        
    def __validate__(self, filename):
        # Perform validation checks on data

        # Verify it has a material ID
        assert 'id' in  self.gen,\
            filename + ": has no valid material id"
        
        # Verify that all cross-sections have the same number of groups
        # by checking that the dimensions are all identical
        if not self.__check_equal__(map(np.shape, self.xsec.values())):
            raise RuntimeError(filename +
                               """: Cross-sections must have the
                               same dimensions""")
        # Verify that all cross-sections are positive
        if not np.all(map(lambda x: x>=0, self.xsec.values())):
            raise RuntimeError(filename +
                               ': contains negative cross-section.')

    ## UTILITY FUNCTIONS =============================================
        
    def __dict_add__(self, dict, el):
        try:
            # Try to convert to float
            val = float(el.text)
        except ValueError:
            try:
                # Try to convert if a comma separated list of floats
                val = np.array(map(float, el.text.split(',')))
            except ValueError:
                # Just store the string
                val = el.text
        dict.update({el.tag: val})
        
    def __check_equal__(self, iterator):
        """ Checks that all entries in iterator are identical
        """
        iterator = iter(iterator)
            
        try:
            first = next(iterator)
        except StopIteration:
            return True
            
        return all(first == rest for rest in iterator)
        
class material(object):
    def __init__(self):
        '''@brief constructor of material class

        @param self The reference to current instance of the material class
        '''
        self.n_materials = 1#change it
        self.n_group = 1#change it
        self.g_thermal = 0#from which we can see upscattering effect
        # The material properties are stored as dictionaries. The keys are material
        # IDs. Dictionary values are numpy arrays of properties for sigt, nu, sigf
        # and
        # The following is the basic material properties
        self.sigt = dict()
        self.nu = dict()
        self.sigf = dict()
        self.sigs = dict()
        self.chi = dict()

        # The following is the derived properties based on basic properties
        self.sigs_per_str = dict()
        self.chi_nu_sigf = dict()
        self.chi_nu_sigf_per_str = dict()
        self.diff_coef = dict()
        self.inv_sigt = dict()

        # The following is for upscattering acceleration
        # spectrum
        self.ksi_ua = dict()
        # sigt one group
        self.sigt_ua = dict()
        # diff_coef one group
        self.diff_coef_ua = dict()
        # TODO: put whatever else necessary parameters if needed        

    def read_xsec(self,xsec_filename):
        """@brief read cross sections from xml file

        @param xsec_filename The xml filename
        """
        # TODO: fill in material properties reading. Please read in the following:
        # self.sigt; self.nu; self.sigf; self.sigs
        # data structures are dictionaries of numpy arraies

    def read_material_id(self,id_filename):
        """@brief read mateiral id

        @param id_filename The xml filename
        """
        # TODO: fill in ID reading

    def derived_properties(self):
        '''@brief derive properties after reading in  basic properties
        '''
        # inv_sigt and diff_coef:
        for k, v in self.sigt.items():
            self.inv_sigt[k] = 1. / self.sigt[k]
            self.diff_coef[k] = 1./ (3. * self.sigt[k])
        # sigs_per_str:
        for k, v in self.sigs.items():
            self.sigs_per_str[k] = v / (4.0 * pi)
        # nu_sigf and nu_sigf_per_str:
        for k in range(self.n_materials):
            self.chi_nu_sigf[k] = np.outer(self.chi[k], self.nu[k]*self.sigf[k])
            self.chi_nu_sigf_per_str[k] = self.chi_nu_sigf[k] / (4.0 * pi)

    def derive_scattering_eigenvalue(self):
        for k, v in self.sigs.items():
            # slicing to get thermal group transfer matrices
            thermal_scat = v[self.g_thermal:, self.g_thermal:]
            self.ksi_ua[k] = np.linalg.eigvals(thermal_scat)

    def produce_acceleration_properties(self):
        for k in xrange(n_materials):
            self.sigt_ua[k] = 0.0
            self.diff_coef_ua[k] = 0.0
            self.sigt_ua[k] = np.dot(self.ksi_ua[k], self.sigt[k][self.g_thermal:])
            self.diff_coef_ua[k] = np.dot(self.ksi_ua[k], self.diff_coef[k][self.g_thermal:])

    def estimate_cell_correction_ua(self, cell_corrections_at_qp, material_id):
        '''@brief function used to estimate one-group vector_D at quadrature points

        @param cell_corrections_at_qp vector_D for all groups at the specific quadrature
        point. Use numpy.array
        @param material_id Materail ID for current cell
        @return a one-group vector_D in forms of numpy array
        '''
        vector_D = np.array([0.0, 0.0])
        # x component of vector_D
        vector_D[0] = np.dot(cell_corrections_at_qp[0][self.g_thermal:], self.ksi_ua[material_id])
        # y component of vector_D
        vector_D[1] = np.dot(cell_corrections_at_qp[1][self.g_thermal:], self.ksi_ua[material_id])
        return vector_D

    def estimate_bd_correction_ua(self, bd_corrections_at_qp, material_id):
        '''@brief function used to estimate one-group kappa at quadrature points

        @param bd_corrections_at_qp kappa for all groups at the specific quadrature
        point.
        @param material_id Materail ID for current cell
        @return a one-group kappa
        '''
        return np.dot(bd_corrections_at_qp, self.ksi_ua[material_id])
