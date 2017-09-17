import subprocess
import sys
import os

JOBSCRIPT = """#!/bin/bash

#$ -S /bin/bash
#$ -N {}
#$ -cwd
#$ -V
#$ -o process_output/
#$ -e process_output/

python -m atsim process {}"""


def main(sid):
    """
    We need a separate script to submit `process.py` since we can't access
    `qstat` from a compute node.

    """

    # First a crude check that the simulation series is not running/queued:
    jid = sid.split('_')[-1]

    print('sid: {}'.format(sid))
    print('jid: {}'.format(jid))

    qstat = subprocess.check_output(['qstat']).decode('utf-8')

    if (' ' + jid + ' ') in qstat:
        print('Found {} in `qstat`; exiting.'.format(jid))
        return

    else:
        print('Submitting process.py serial job.')

        # Write a jobscript:
        nm = 'p_' + jid

        global JOBSCRIPT
        JOBSCRIPT = JOBSCRIPT.format(nm, sid)

        js_fn = 'js_temp.sh'
        with open(js_fn, 'w', newline='') as jsf:
            jsf.write(JOBSCRIPT)

        # Submit the jobscript:
        subprocess.run(['qsub', js_fn])

        # Remove the jobscript:
        os.remove(js_fn)
