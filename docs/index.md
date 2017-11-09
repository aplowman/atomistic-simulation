---
source: ['makesims_opt.yml']
---

# Options for generating simulations
## method
`"lammps" | "castep"`

Sets the modelling software to use.

## stage
Sets options which detemine where on the local machine the simulations should be generated
### stage.path
`string`

The absolute path on the local machine in which the simulaton series directory will be generated.

## database
Sets options for access to the database.
### database.dropbox
`boolean`

If `True`, the database is assumed to be located (or will be generated) on Dropbox. If False, the database is assumed to be located on the local machine.

## check
For each `check`, if "all", the check is performed after the generation of both the base structure and the series structures. If "base", the check is performed after the generation of the base structure only. If "series", the check is performed after the generation of each series structure. If "none", the check is not performed at all.
### check.atoms_overlap
`"all" | "base" | "series" | "none"`

`default: "none"`

Determines when to check if any atoms are closer than the distance defined in `base_structure.overlap_tol`. 

### check.bicrystal_inversion_symmetry
`"all" | "base" | "series" | "none"`

`default: "none"`

Determines when to check if the generated bicrystal exhibits inversion symmetry through the centre of its one crystals. 


<!-- 
# Example options

{% for c in page.source %}

  {% capture filePath %}{{c}}{% endcapture %}

  <a href="{{filePath}}">{{c}}</a>

  ```yaml
{% include_relative {{ filePath }} %}
  ```
{% endfor %}

# Simulation series

## Series definition dicts
A simulation series is defined by a series dict which contains at least a `name` key. Many series types also support a `vals` key. For example:

<pre>
{
    'name': 'cut_off_energy',
    'vals': [250, 300, 350]
}
</pre>
Here the dict represents a cut off energy simulation series with three simulations, each one using a different cut off energy.

Most series definition dicts which accept a `vals` key alternatively accept keys `start`, `step` and `stop`. For example, we could equivalently define the same cut off energy series as:
<pre>
{
    'name': 'cut_off_energy',
    'start': 250,
    'step': 50,
    'stop': 350,
}
</pre>
Note that `start` and `stop` are inclusive.

## Supported series types
Below are some example series definition dicts.
### CASTEP or LAMMPS
**Grain boundary expansion**
<pre>
{
    'name': 'boundary_vac',
    'vals': [0.1, 0.2],
}
</pre>
**γ-surface**
<pre>
{
    'name': 'gamma_surface',
    'grid_spec': {
        'size': (3, 4),
    },
}
</pre>
### CASTEP specific
**Cut off energy**
<pre>
{
    'name': 'cut_off_energy',
    'vals': [250, 300],
}
</pre>
**K-point spacing**
<pre>
{
    'name': 'kpoint',
    'vals': [0.05, 0.04],
}
</pre>
## Combining series definitions
The `series` object in the options file controls the setting up of a simulation series. `series` is a list of lists of dicts. Outer list elements represent simulation series which are to be nested. Inner list elements represent parallel simulation series which are combined. Within each inner list is one or more series definition dicts. For example:

<pre>
series = [
    [
        {
            'name': 'cut_off_energy',
            'vals': [250, 300, 350],
        }
    ],
    [
        {
            'name': 'kpoint',
            'vals': [0.07, 0.08],
        }
    ]
]
</pre>

In this case, we set up a simulation series to test three cut off energy values and for each cut off energy, we test two kpoint values (for a total of six simulations). In some cases it's useful to generate parallel series:

<pre>
series = [
    [
        {
            'name': 'cut_off_energy',
            'vals': [250, 300],
        },
        {
            'name': 'kpoint',
            'vals': [0.07, 0.08],
            ]
        }
    ]
]
</pre>

Note that here the cut off energy and kpoint series dicts are in the same inner list. This will results in two simulations: the first with a cut off energy of 250 and a kpoint spacing of 0.07; the second with a cut off energy of 300 and a kpoint spacing of 0.08.

## Example series
### γ-surface and boundary expansion

<pre>
series = [
    [
        {
            'name': 'boundary_vac',
            'start': -0.2,
            'step': 0.1,
            'stop': 0.4,
        }
    ],
    [
        {
            'name': 'gamma_surface',
            'grid_spec': {
                'size': (4, 5),
            }
        }
    ]
]
</pre>

# Harvesting results
Collating results is done by running `harvest.py`. Options for harvesting results are specified in a Python file, `harvest_opt.py` (in the `set_up` directory), which contains a single dict named `HARVEST_OPT`. Running `harvest.py` generates a JSON file, `results.json`, and optionally generates plots of the data listed in the JSON file.

## Keys in `HARVEST_OPT`
<dl>
    <dt><code>sid</code> : list of str</dt>
    <dd>
        Lists which simulation series are to be included in the results harvesting.
    </dd>
    <dt><code>overwrite_results</code> : bool or str (True | False | 'ask'), optional</dt>
    <dd>
        The results for each simulation are stored in the <code>results</code> attribute of the <code>AtomisticSimulation</code> object for that simulation. If <code>harvest.py</code> encounters a simulation for which results have already been assigned, this boolean determines whether to overwrite them or not.
    </dd>
    <dt><code>debug</code> : bool, optional</dt>
    <dd>
        If True, <code>results.json</code> and plots are generated inside a directory named <code>0000-00-00-0000-00000</code>. If False, a new results ID is generated in the format: <code>YYYY-MM-DD-TTTT-RRRRR</code> where Y, M, D, T refer to the current year, month, day and time respectively, and R are random digits. <code>results.json</code> and plots are then placed inside a directory with this name.
    </dd>
    <dt><code>skip_idx</code> : list of list of int, optional</dt>
    <dd>
        a
    </dd>
    <dt><code>variables</code> : list of dict, optional</dt>
    <dd>
        a
    </dd>
    <dt><code>plots</code> : list of dict, optional</dt>
    <dd>
        a
    </dd>
</dl>

## Specifying simulaton series
Simulation series identifiers are listed in the `HARVEST_OPT` key `sid`. To skip some simulations, specify their indices within their series in the key `skip_idx`. For example, let's say we want to collate the results from three simulation series, but the first two simulations of the second series failed and so we don't want to include them. We would specify the options like this:
<pre>
HARVEST_OPT = {
    'sid': [
        '2017-08-09-1122_13068',
        '2017-08-09-1137_99144',
        '2017-08-09-1125_28108',
    ],
    'skip_idx': [
        [],
        [0, 1],
        [],
    ]
}
</pre>

## Variables
Each variable is represented as a dict inside the `variables` list. Each variables must have the keys: `type`, `id`, and `name`. Allowed `type` values are:
* `result`
* `parameter`
* `compute`
* `series_id`

The variable `id` is chosen as a unique identifier for the variable. This then allows us to reference the variable later on (for instance in the data for a plot). -->
