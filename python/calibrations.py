"""
Manager of the calibration data.

This module provides the code managing the calibration data used by the collections.

Classes:
    CalibManager

"""
import json
import os
import errno
import numpy as np
import pandas as pd

from . import selections


class CalibManager:
    """
    CalibManager.

    Manages the calibration data ensuring coherent versioning, reading from json
    and coherently serving them to the rest of the code when/if needed.
    
    Reads generic dictionaries from JSON files and maps them to a key for retrieval 
    in the rest of the code.

    It is a singleton.
    """

    class __TheManager:
        def __init__(self):
            self.calib_files = {}

        def set_calib_file(self, calib_name, file_name):
            pwd = os.path.dirname(__file__)
            rel_filename = os.path.join(pwd, '..', file_name)
            if os.path.isfile(rel_filename):
                self.calib_files[calib_name] = rel_filename
            else:
                raise FileNotFoundError(
                    errno.ENOENT, os.strerror(errno.ENOENT), rel_filename)

        def get_calib(self, calib_name):
            # FIXME: could cache the JSON data
            calibs = {}
            with open(self.calib_files[calib_name]) as f:
                calibs = json.load(f)
            return calibs

        def get_calibration(self, collection_name, calib_key):
            print('WARNING: call to get_calibration is DEPRECATED, please use get_calibs("hgc_empt_calib")["collection_name"]["calib_key"]')
            return self.get_calibs('hgc_empt_calib')[collection_name][calib_key]

        def get_pt_wps(self):
            print('WARNING: call to get_pt_wps is DEPRECATED, please use get_calibs("rate_pt_wps")')
            return self.get_calibs('rate_pt_wps')
            
    instance = None

    def __new__(cls):
        if not CalibManager.instance:
            CalibManager.instance = CalibManager.__TheManager()
        return CalibManager.instance

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def __setattr__(self, name):
        return setattr(self.instance, name)


