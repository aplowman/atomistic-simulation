"""matsim.atomistic.structure.__init__.py"""

import os
from pathlib import Path
import numpy as np


def site_labs_to_jsonable(labs):
    """Get a representation of a site labels dict which can be JSON
    serialised.

    Site labels have the form dict of {str: tuple of ndarrays}

    """
    if labs is None:
        return
    labs_js = {}
    for lab_name, lab_val in labs.items():
        labs_js.update({
            lab_name: [np.array(i).tolist() for i in lab_val]
        })
    return labs_js


def site_labs_from_jsonable(labs):
    """Convert site labels from JSONable form to desired form."""
    if labs is None:
        return
    labs_native = {}
    for lab_name, lab_val in labs.items():
        labs_native.update({
            lab_name: tuple([np.array(i) for i in lab_val])
        })
    return labs_native
