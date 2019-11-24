# Stochastic Triangular Mesh (STM) Mapping

An online dense mapping technique for mobile robots.

## Dependencies

### Required

* `git`
* `python` >= 3.7
* `EMDW`---a proprietary PGM library, which is in the process of being made open source

### Recommended

* `pyenv`
* `pyenv-virtualenv`

## PyEMDW Wrapper Library

PyEMDW is a python wrapper for some of the functionality of EMDW. The release of
PyEMDW is pending the EMDW open source release.

## STM Mapping Library

The STM mapping library is under the `stm-map` directory.  Assuming PyEMDW
is installed, to install the library, from the root directory, run

`$ pip install ./stm-map`

## Experiments

This contains a series of experiments testing the STM mapping technique. From
within the `experiments` directory run a desired script as a module---that is,

`$ python -m folder.script {kwargs}`

For example

`$ python -m compare_surface_maps.elevation_compare_2d`

or

`$ python -m utoronto.main ~/dataset_directory --batch`
