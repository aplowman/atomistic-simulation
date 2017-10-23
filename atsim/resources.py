import utils


class SGEOptions(object):
    """
    Class to represent options for sumitting a simulation series to a SGE batch
    scheduler.

    """

    def __init__(self, job_array, selective_submission=False, job_name=None,
                 parallel_env=None, resource=None):
        """
        Parameters
        ----------
        job_array : bool
            If True, calculations are submitted as an SGE job array. If
            False, calculations are submitted in one go. If the number
            of calculations is one, this will be set to False. Setting
            this to False can be handy for many small calculations
            which won't take a long time to complete. If submitted as
            a job array, a significant fraction of the total completion
            time may be queuing.
        selective_submission : bool, optional
            If True, the SGE task id flag `-t` [1] will be excluded
            from the jobscript file and instead this flag will be
            expected as a command line argument when executing the
            jobscript: e.g. "qsub -t 1-10:2 jobscript.sh". Default is
            False.
        job_name : str, optional
            Default is None.
        parallel_env : str, optional
            The SGE parallel environment on which to submit the calculations.
            Only applicable in `num_cores` > 1. Default is None.
        resource : str, optional
            The value to set for the SGE resource flag "-l". Default is None,
            in which case the "-l" flag is not set.

        References
        ----------
        [1] http://gridscheduler.sourceforge.net/htmlman/htmlman1/qsub.html


        """


class ComputeOptions(object):
    """Class to represent the options used to submit a simulation series."""

    def __init__(self, scratch, num_cores, sge=None, module_load=None):
        """
        Parameters
        ----------
        num_cores : int
            Number of processor cores to use.
        sge : SGEOptions object, optional
            Options for submitting the simulation series to a SGE batch
            scheduler. Default is None, in which case job is assumed not to be 
            scheduled.

        """

        self.scratch = scratch
        self.num_cores = num_cores
        self.sge = sge
        self.module_load = module_load


class Resource(object):

    """Class to represent a directory on a remote or local computer."""

    def __init__(self, is_dropbox=False, host=None, os_name='posix', path='', remote=False):
        """
        Parameters
        ----------
        is_dropbox : bool, optional
            Specifies whether this resource resides on a Dropbox account.
            Default is False.
        host : str, optional
            Only applicable if `remote` is True. Host name used to connect to
            the remote resource using SSH. Default is None. Must be specified
            if `remote` is True.
        os_name : str, optional
            One of "posix" (for MacOS and Unix-like machines) or "nt" (for 
            Windows). Default is "posix".
        path : str, optional
            Directory path of the resource. Set to empty string by default.
        remote : bool, optional
            Specifies whether the resource is on the local or a remote machine.
            Default is False.

        """

        if remote and not host:
            raise ValueError('`host` must be specified if `remote` is True.')

        elif not remote and host:
            raise ValueError('`host` must not be specified if `remote` is '
                             'False')

        if remote and os_name == 'nt':
            raise NotImplementedError('Remote Windows resource not currently '
                                      'supported.')

        self.is_dropbox = is_dropbox
        self.host = host
        self.os_name = os_name
        self.path = path
        self.remote = remote

        check_access()

    def to_jsonable(self):
        pass

    @classmethod
    def from_jsonable(cls):
        pass

    def check_access(self):
        """Check the resource is accessible."""

        # If on Windows, check we have access to bash
        if self.os_name == 'posix':
            if not utils.check_bash():
                raise NotImplementedError('Cannot find Bash.')

        # Check that self.path is a directory as well

        pass

    def copy_to(self):
        """Copy a directory to this resource."""
        pass

    def copy_from(self):
        """Copy a directory from this resource."""
        pass


class Stage(Resource):
    pass


class Scratch(Resource):
    pass


class Archive(Resource):
    pass
