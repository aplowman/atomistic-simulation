class SGEOptions(object):
    """
    Class to represent options for sumitting a simulation series to a SGE batch
    scheduler.

    """

    def __init__(self, job_array, selective_submission=False, job_name=None,
                 parallel_env=None):
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
            jobscript: e.g. "qsub jobscript.sh -t 1-10:2". Default is
            False.
        job_name : str, optional
            Default is None.
        parallel_env : str, optional
            The SGE parallel environment on which to submit the calculations. Only
            applicable in `num_cores` > 1. Default is None.

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

        check_access()

    def to_jsonable(self):
        pass

    @classmethod
    def from_jsonable(cls):
        pass

    def check_access(self):
        """Check the resource is accessible."""
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
