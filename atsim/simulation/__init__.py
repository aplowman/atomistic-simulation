"""matsim.simulation.__init__.py"""

from collections import namedtuple

BaseUpdate = namedtuple(
    'BaseUpdate',
    ['address', 'val', 'val_seq_type', 'mode']
)
