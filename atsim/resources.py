import os
import pathlib
import subprocess
import yaml
from shutil import copytree
from atsim import SET_UP_PATH, CONFIG
from atsim import utils, database
from atsim.utils import prt, dict_from_list, mut_exc_args


class Resource(object):
    """Class to represent a location on the local or a remote machine."""

    def __init__(self, resource_id, add_path=None):
        """Initialise resource object from ID and 'database' file."""

        res_defn = database.get_resource(resource_id)

        if add_path is None:
            add_path = []

        # Instantiate a "pure" path object in this case, which does not have
        # access to the file system.
        if res_defn['machine_os_type'] == 'nt':
            path_class = pathlib.PureWindowsPath

        elif res_defn['machine_os_type'] == 'posix':
            path_class = pathlib.PurePosixPath

        self.base_path = path_class(res_defn['resource_base_path'])
        self.path = self.base_path.joinpath(*add_path)

        # Check base_path is absolute
        if not self.base_path.is_absolute():
            msg = ('Resource `base_path` "{}" must be an absolute path.')
            raise ValueError(msg.format(self.base_path))

        self.resource_id = res_defn['resource_id']
        self.machine_id = res_defn['machine_id']
        self.machine_name = res_defn['machine_name']
        self.os_type = res_defn['machine_os_type']
        self.is_dropbox = res_defn['machine_is_dropbox']

    def make_paths_concrete(self):
        """Convert `base_path` and `path` to "concrete" path objects.

        Convert to "concrete" path objects (as opposed to "pure") to allow
        file operations.

        """

        if self.os_type == 'nt':
            path_class = pathlib.WindowsPath

        elif self.os_type == 'posix':
            path_class = pathlib.PosixPath

        self.base_path = path_class(self.base_path)
        self.path = path_class(self.path)


class Stage(Resource):
    """Class to represent the area on the local machine in which simulations
    input files are generated."""

    def __init__(self, name, add_path=None):

        stage_defn = database.get_stage_by_name(name)
        res_id = stage_defn['resource_id']
        super().__init__(res_id, add_path)
        self.name = name
        self.stage_id = stage_defn['stage_id']
        self.make_paths_concrete()


class Scratch(Resource):
    """Class to represent the area on a machine in which simulations are to be
    run."""

    def __init__(self, name, add_path=None):

        scratch_defn = database.get_scratch_by_name(name)
        res_id = scratch_defn['resource_id']
        super().__init__(res_id, add_path)
        self.name = name
        self.scratch_id = scratch_defn['scratch_id']
        self.sge = scratch_defn['scratch_is_sge']


class Archive(Resource):
    """Class to represent the area on a machine in which completed simulations
    are archived."""

    def __init__(self, name, add_path=None):

        arch_defn = database.get_archive_by_name(name)
        res_id = arch_defn['resource_id']
        super().__init__(res_id, add_path)
        self.name = name
        self.archive_id = arch_defn['archive_id']


class ResourceConnection(object):
    """Class to represent a connection between a local resource and another."""

    def __init__(self, src, dst):

        res_conn = database.get_resource_connection(
            src.resource_id, dst.resource_id)

        if res_conn is None:
            raise ValueError('No resource connection information between source'
                             'and destination resources can be found.')

        self.src = src
        self.dst = dst
        self.host = res_conn['host']
        self.remote = src.machine_id != dst.machine_id
        self.os_types = (self.src.os_type, self.dst.os_type)

        if self.remote:

            # These (src, dst) OS types are allowed for remote dst:
            ok_remote_os_types = {
                ('nt', 'posix'),
                ('posix', 'posix'),
            }
            if self.os_types not in ok_remote_os_types:
                msg = ('This combination of source and destination '
                       '`os_type` {} is not supported for remote destination.')
                raise ValueError(msg.format(self.os_types))

        # Source is always local:
        self.src.make_paths_concrete()

        # Destination may be remote:
        if not self.remote:
            self.dst.make_paths_concrete()

    def check_conn(self):
        """Check a connection can be made between source and destination
        resources, and that the base paths of both source and destination exist.

        TODO: separate remote connection and destination base path check.

        """
        # Check source base path exists:
        if not self.src.base_path.exists():
            msg = 'Source `base_path` "" does not exist.'
            raise ValueError(msg.format(self.src.base_path))

        if self.remote:

            # Check remote connection can be made and destination base path exists:
            ssh_cm = 'ssh {} "[ -d {} ]"'.format(self.host, self.dst.base_path)
            comp_proc = subprocess.run(['bash', '-c', ssh_cm])

            if comp_proc.returncode == 1:
                msg = ('Remote connection to host "{}" could not be made, or '
                       'destination `base_path` "{}" does not exist')
                raise ValueError(msg.format(self.host, self.dst.base_path))

        else:

            # Check destination base path exists:
            if not self.dst.base_path.exists():
                msg = 'Destination `base_path` "" does not exist.'
                raise ValueError(msg.format(self.dst.base_path))

    def copy_from_dest(self, path):
        """Copy content from the destination resource to the source resource.

        """
        self.check_conn()

        msg = ('Copying from resource "{}" to{} resource "{}".')
        is_rem_str = ' remote' if self.remote else ''
        print(msg.format(self.dst.name, is_rem_str, self.src.name))

        # if self.remote:

        #     if self.src.os_type == 'nt':

        #         # Convert path to posix style for use within "Bash on Windows":
        #         path_args = ['/mnt', self.src.path.drive[0].lower(),
        #                      *self.src.path.parts[1:]]
        #         src_path = pathlib.PurePosixPath(*path_args)

        #     elif self.src.os_type == 'posix':
        #         src_path = self.src.path

        #     # Add trailing slash to path (for rsync):
        #     src_path = str(src_path) + '/'
        #     utils.rsync_remote(src_path, self.host, self.dst.path, mkdirs=True)

        # else:

        #     if self.dst.path.exists():
        #         raise ValueError(exists_msg)

        #     copytree(self.src.path, self.dst.path)

    def copy_to_dest(self):
        """Copy content from the source resource to the destination resource.

        """
        exists_msg = 'Destination directory already exists.'
        self.check_conn()

        msg = ('Copying from resource "{}" to{} resource "{}".')
        is_rem_str = ' remote' if self.remote else ''
        print(msg.format(self.src.name, is_rem_str, self.dst.name))

        if self.remote:

            if self.src.os_type == 'nt':

                # Convert path to posix style for use within "Bash on Windows":
                path_args = ['/mnt', self.src.path.drive[0].lower(),
                             *self.src.path.parts[1:]]
                src_path = pathlib.PurePosixPath(*path_args)

            elif self.src.os_type == 'posix':
                src_path = self.src.path

            # Add trailing slash to path (for rsync):
            src_path = str(src_path) + '/'
            utils.rsync_remote(src_path, self.host, self.dst.path, mkdirs=True)

        else:

            if self.dst.path.exists():
                raise ValueError(exists_msg)

            copytree(self.src.path, self.dst.path)

    def run_command(self, cmd, cwd=None):
        """Execute a command on the destination resource.

        Parameters
        ----------
        cmd : str
            The command to execute
        cwd : str, optional
            The directory in which to execute the command, relative to the
            `path` of the destination resource.

        """

        self.check_conn()

        # no_dir_msg = 'Directory does not exist on scratch. Aborting.'

        if cwd is not None:
            cwd = str(self.dst.path.joinpath(cwd))
        else:
            cwd = str(self.dst.path)

        if self.remote:

            run_args = ['bash', '-c']
            cmd_str = 'ssh {} "cd {} && {}"'
            run_args.append(cmd_str.format(self.host, cwd, cmd))
            _ = subprocess.run(run_args)

        else:

            if self.dst.os_type == 'nt':
                subprocess.run([cmd], shell=True, cwd=cwd)

            elif self.dst.os_type == 'posix':
                pass

        #     if self.os_name == 'nt' and scratch.os_name == 'nt':
        #         if not os.path.isdir(scratch.path):
        #             raise ValueError(no_dir_msg)

        #         js_path = os.path.join(scratch.path, 'jobscript.bat')
        #         # Run batch file in a new console window:
        #         subprocess.Popen(js_path,
        #                          creationflags=subprocess.CREATE_NEW_CONSOLE)

        #     elif self.os_name == 'posix' and scratch.os_name == 'posix':
        #         # Use rsync/scp
        #         js_path = os.path.join(scratch.path, 'jobscript.sh')
        #         os.chmod(js_path, 0o744)
        #         subprocess.Popen(js_path, shell=True)
