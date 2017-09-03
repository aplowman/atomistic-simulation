# atomistic-simulation

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
            'vals': [
                250,
                300,
                350
            ]
        }
    ],
    [
        {
            'name': 'kpoint',
            'vals': [
                0.07,
                0.08
            ]
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
            'vals': [
                250,
                300
            ]
        },
        {
            'name': 'kpoint',
            'vals': [
                0.07,
                0.08
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
