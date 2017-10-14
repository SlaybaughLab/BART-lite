import numpy as np
import warnings
import re
from math import pi
import os, sys
import xml.etree.cElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class _mat():
    def __init__(self, filename, grps, tr_scatt=False):
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

        self.n_grps = grps
        self.gen = {}  # General properties
        self.prop = {}  # Physical properties
        self.gconst = {}  # Group non cross-section data
        self.xsec = {}  # Group cross-section data
        self.derived = {}  # Derived quantities

        # All dictionaries that hold material parameters
        self.all_dict = [
            self.gen, self.prop, self.gconst, self.xsec, self.derived
        ]

        self.__parse_XML__(filename, grps)  # Parse input XML file
        self.__validate__(filename)  # Validate material data

        # If needed, transpose scattering matrix
        if 'sig_s' in self.xsec and tr_scatt:
            self.xsec['sig_s'] = np.transpose(self.xsec['sig_s'])

        if self.xsec:
            self.__derive_xsec__()  # Calc other xsecs

        if 'sig_t' in self.xsec:
            self.__derive_sig_t__()  # Calc sig_t derv. prop

        if 'sig_s' in self.xsec and 'g_thermal' in self.gconst:
            self.__derive_thermal__()  # Calc thermal eigenvalues

        if self.isSource:
            self.__derive_fiss__()  # Calc fission derv. prop

        if 'sig_t' in self.xsec and 'ksi_ua' in self.derived and\
           'g_thermal' in self.gconst:
            self.__derive_acceleration__()  # Calc acceleration

    # PUBLIC FUNCTIONS

    def get_props(self):
        # Returns an array of all the properties that the material has
        return [d.keys() for d in self.all_dict]

    def get(self, prop):
        # Returns the value of a given property prop
        try:
            return next(d[prop] for d in self.all_dict if prop in d)
        except StopIteration:
            raise RuntimeError("Invalid material property for " +
                               self.gen['id'] + ": " + prop)

    # INITIALIZATION FUNCTIONS ========================================

    def __derive_acceleration__(self):
        # Calculate acceleration properties
        i = int(self.gconst['g_thermal'])

        sig_t_ua = np.dot(self.derived['ksi_ua'], self.xsec['sig_t'][i:])

        diff_coef_ua = np.dot(self.derived['ksi_ua'],
                              self.derived['diff_coef'][i:])

        try:
            sig_r_ua = sig_t_ua - \
                       np.sum(np.multiply(self.derived['ksi_ua'],
                                          self.xsec['sig_s'][i:,i:]))
            self.derived.update({'sig_r_ua': sig_r_ua})
        except KeyError:
            pass

        self.derived.update({'sig_t_ua': sig_t_ua})
        self.derived.update({'diff_coef_ua': diff_coef_ua})

    def __derive_sig_t__(self):
        # Calculate derived quantities based on sig_t

        # Find non-zero entries of Sig_t
        non_zero = self.xsec['sig_t'] != 0
        # Inverse Sig_t
        inv_sig_t = np.power(self.xsec['sig_t'], -1, where=non_zero)
        # Diffusion Coeff
        diff_coef = np.power(3. * self.xsec['sig_t'], -1, where=non_zero)

        self.derived.update({'diff_coef': diff_coef})
        self.derived.update({'inv_sig_t': inv_sig_t})

    def __derive_fiss__(self):
        # Calculate derived quantities based on fission properties

        if 'chi' in self.gconst and 'sig_f' in self.xsec:
            vec_2 = self.prop['nu'] * self.xsec['sig_f']
        elif 'nu_sig_f' in self.xsec:
            vec_2 = self.xsec['nu_sig_f']
        elif 'nu_sig_f' in self.gconst:
            vec_2 = self.gconst['nu_sig_f']
        else:
            vec_2 = np.array([0, 0])

        chi_nu_sig_f = np.outer(self.gconst['chi'], vec_2)

        self.derived.update({'chi_nu_sig_f': chi_nu_sig_f})

    def __derive_thermal__(self):
        # Calculate quantities based on thermal scattering
        i = int(self.gconst['g_thermal'])

        # Slice scattering matrix based on g_thermal
        thermal = self.xsec['sig_s'][i:, i:]
        th_d_i = np.tril(thermal)
        th_u = np.triu(thermal, 1)

        total = np.diag(self.xsec['sig_t'][i:])
        try:
            M = np.matmul(np.linalg.inv(total - th_d_i), th_u)
            w, v = np.linalg.eig(M)
            ksi_ua = v[:, np.argmax(np.absolute(w))]
            ksi_ua = ksi_ua / np.sum(ksi_ua)
        except np.linalg.LinAlgError:
            warnings.warn("Matrix for thermal eigenvalue is singular," +
                          "setting value of ksi_ua to 0")
            ksi_ua = np.zeros(self.n_grps - i)

        self.derived.update({'ksi_ua': ksi_ua})

    def __derive_xsec__(self):
        if 'sig_t' in self.xsec and 'sig_s' in self.xsec:
            sig_r = self.xsec['sig_t'] - np.diag(self.xsec['sig_s'])
            self.xsec.update({'sig_r': sig_r})

    def __parse_XML__(self, filename, grps):
        # Parse the XML file

        # These are the tags that identify the different sections of
        # the XML file. They are here to make it easy to change it in
        # the future.
        tag_prop = "prop"  # Phys. Properties
        tag_grp_structures = "grp_structures"  # Group structures
        tag_xsec = "xsec"  # Cross-sections

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
            grp_root = grp_structs.findall(".*[@n='" + str(self.n_grps) +
                                           "']")[0]
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
        elif 'nu_sig_f' in self.xsec or 'nu_sig_f' in self.gconst:
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

        for key, a in self.xsec.iteritems():
            if np.shape(a) != (self.n_grps,) and\
               np.shape(a) != (self.n_grps, self.n_grps):
                raise RuntimeError(filename + """: Cross-sections must have the
                               same dimensions, error with: """ + key)

        # Verify that all cross-sections are positive
        if not all(
            [np.all(m) for m in map(lambda x: x >= 0, self.xsec.values())]):
            raise RuntimeError(filename + ': contains negative cross-section.')

        # Verify that g_thermal is not higher than the number of groups
        if 'g_thermal' in self.gconst:
            if self.gconst['g_thermal'] + 1 > self.n_grps:
                raise RuntimeError(filename + ': g_thermal > n_groups')
            if not self.gconst['g_thermal'].is_integer():
                raise RuntimeError(filename + ': g_thermal must be an integer')

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
                try:
                    # Try to convert to a matrix
                    val = np.array(
                        [map(float, s.split(',')) for s in el.text.split(';')])
                # Just store the string
                except ValueError:
                    val = el.text
        dict.update({el.tag: val})


class mat_lib():
    def __init__(self, n_grps, files=[], tr_scatt=False):
        '''Material Library class, holds multiple _mat objects provided
        at initialization or added later.

        files: list of filenames to material xml files
        '''
        self.mats = []  # Holds all materials
        self._n_grps = n_grps  # Energy groups

        for f in files:
            self.add(f, tr_scatt)

    def add(self, filename, tr_scatt=False):
        """ Adds the material stored in filename to the library, if it
        is not already in there. """

        new_mat = _mat(filename, grps=self._n_grps, tr_scatt=tr_scatt)

        if new_mat.gen['id'] not in self.ids():
            self.mats.append(new_mat)
        else:
            raise RuntimeError("Cannot add file " + filename +
                               ", mat_id already exists in material library")

    def n_grps(self):
        return self._n_grps

    def ids(self):
        """ Returns the id's of stored materials """
        return [mat.gen['id'] for mat in self.mats]

    def get(self, prop, mat_id=''):
        """ Returns a dictionary with material ids as keys and the
        specified property as values"""

        if prop == 'n_grps':
            return self._n_grps

        data = self.__mat_data__(prop)

        if mat_id:
            try:
                return data[mat_id]
            except KeyError:
                raise KeyError("Bad material id")
        else:
            return data

    def get_per_str(self, *args, **kwargs):
        try:
            return np.divide(self.get(*args, **kwargs), 4.0 * np.pi)
        except TypeError:
            return {
                k: np.divide(v, 4.0 * np.pi)
                for k, v in self.get(*args, **kwargs).iteritems()
            }

    def props(self, mat_id=None):
        data = {}
        for mat in self.mats:
            data.update({mat.gen['id']: mat.get_props()})
        if mat_id:
            return data[mat_id]
        else:
            return data

    def __mat_data__(self, prop):
        data = {}

        for mat in self.mats:
            data.update({mat.get('id'): mat.get(prop)})

        return data


class mat_map():
    def __init__(self,
                 lib,
                 layout,
                 layout_dict,
                 x_max,
                 n,
                 x_min=0,
                 y_min=0,
                 y_max=None):
        """ mat map will create a material map based on a string input
        map and problem parameters """
        x = [x_min, x_max]
        y = [y_min, y_max] if y_max else [y_min, x_max]

        self.mat_dict = layout_dict
        self.mat_lib = lib

        try:
            self.x = map(float, x)
            self.y = map(float, y)
        except ValueError:
            raise ValueError("x and y domain limits must be numbers")

        self.dx = x[1] / float(n)
        self.dy = y[1] / float(n)
        self.n = int(n)

        #Generate layout
        # Split into words
        split_layout = re.sub("[^\w]", " ", layout).split()

        # Verify a square number have been given
        n_dim = np.sqrt(len(split_layout))

        assert n_dim.is_integer(),\
            "Layout must have a square number of entries"

        n_dim = int(n_dim)

        self.layout = [
            split_layout[i:i + n_dim]
            for i in range(0, len(split_layout), n_dim)
        ]

        self.array = self.__build_array__()

    def plot(self):  # pragma: no cover
        n = int(np.sqrt(len(self.array)))
        mat_set = list(set(self.array))
        layout = [self.array[x:x + n] for x in range(0, len(self.array), n)]
        for j, s in enumerate(mat_set):
            for i, row in enumerate(layout):
                layout[i] = [r.replace(s, str(j * 10)) for r in row]
        fl_array = np.flipud(np.array([map(float, row) for row in layout]))

        plt.figure(figsize=(6, 6))
        values = np.unique(fl_array)
        im = plt.imshow(fl_array, interpolation='none', extent=[0, n, 0, n])
        colors = [im.cmap(im.norm(value)) for value in values]
        # create a patch (proxy artist) for every color
        patches = [
            mpatches.Patch(
                color=colors[i],
                label="{l}".format(l=mat_set[int(values[i] / 10.0)]))
            for i in range(len(values))
        ]
        # put those patched as legend-handles into the legend
        plt.legend(
            handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        plt.grid(True)
        plt.show()

    def get(self, prop, loc):
        # Get property from material at given location, loc is either
        # the index of the location k or a tuple of x and y
        if isinstance(loc, tuple):
            k = int(loc[0] / self.dx) + int(loc[1] / self.dy) * self.n
        else:
            k = loc

        return self.mat_lib.get(prop=prop, mat_id=self.array[k])

    def __build_array__(self):
        # Builds the array
        try:
            array = []
            for row in reversed(self.layout):
                to_add = []
                for col in row:
                    to_add += int(
                        1.0 / len(row) * self.n) * [self.mat_dict[col]]
                to_add = int(1.0 / len(row) * self.n) * to_add
                array += to_add
            return array
        except KeyError:
            raise KeyError("Bad material id in mat_dictionary")
