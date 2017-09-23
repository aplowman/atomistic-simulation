from setuptools import find_packages, setup

print(find_packages())

setup(name='atsim',
      version="0.1",
      description="Python tools for setting up and analysing the results from CASTEP and LAMMPS simulations.",
      author='APlowman',
      packages=find_packages(),
      install_requires=[
          'plotly',
          'spglib',
          'numpy',
          'matplotlib',
          'mendeleev',
          'dropbox',
          'bokeh',
          'PyYAML',
      ],
      entry_points={
          'console_scripts': []
      }
      )
