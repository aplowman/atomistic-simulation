import os
import yaml
from atsim import SET_UP_PATH, OPT_FILE_NAMES
from atsim import utils
from atsim.utils import prt, dict_from_list

with open(os.path.join(SET_UP_PATH, OPT_FILE_NAMES['resources'])) as res_fs:
    RESOURCES = yaml.safe_load(res_fs)


class Resource(object):
    """Class to represent a location on the local or a remote machine."""

    def __init__(self, resource_id):
        """Initialise resource object from ID and 'database' file."""
        res = dict_from_list(RESOURCES['resource'], {'id': resource_id})
        machine = dict_from_list(RESOURCES['machine'], {
                                 'id': res['machine_id']})
        self.resource_id = resource_id
        self.machine_id = res['machine_id']
        self.name = res['name']
        self.path = res['path']
        self.os_type = machine['os_type']
        self.is_dropbox = machine['is_dropbox']


class Stage(Resource):
    """Class to represent the area on the local machine in which simulations
    input files are generated."""

    def __init__(self, stage_id):

        stage = dict_from_list(RESOURCES['stage'], {'id': stage_id})
        res_id = stage['resource_id']
        super().__init__(res_id)


class Scratch(Resource):
    """Class to represent the area on a machine in which simulations are to be
    run."""

    def __init__(self, scratch_id):

        scratch = dict_from_list(RESOURCES['scratch'], {'id': scratch_id})
        res_id = scratch['resource_id']
        super().__init__(res_id)


class Archive(Resource):
    """Class to represent the area on a machine in which completed simulations
    are archived."""

    def __init__(self, archive_id):

        archive = dict_from_list(RESOURCES['archive'], {'id': archive_id})
        res_id = archive['resource_id']
        super().__init__(res_id)


class ResourceConnection(object):
    """Class to represent a connection between two resources."""

    def __init__(self, src, dst):

        res_conn = dict_from_list(
            RESOURCES['resource_conn'],
            {'source_id': src.resource_id,
             'destination_id': dst.resource_id}
        )

        if res_conn is None:
            raise ValueError('No resource connection between source and '
                             'destination resources can be found.')

        self.source = src
        self.destination = dst
        self.host = res_conn['host']
        self.remote = src.machine_id != dst.machine_id

    def check(self):
        """Check a connection can be made between source and destination
        resources.
        """
        pass

    def copy(self):
        """Copy the contents of the source resource to the destination
        resource.

        """

        pass
