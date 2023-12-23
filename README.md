# A collection of Hamiltonian systems
These include

1. the harmonic oscillator,
2. the simple pendulum,
3. the wave equation (discretized with Finite Differences).

## Installation (Linux)
To guarantee exact reproduction of the code (a) install Python3.7 and (b) install the package (and dependencies) via pip.

### Installation of python3.7 and virtual environment
Install python3.7, e.g. in Ubuntu

```
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.7 -y
sudo apt install python3.7-venv -y
sudo apt-get install python3.7-tk -y
```

Generate a virtual environment in python3.7

```
python3.7 -m venv ~/.venvs/experiments-manifold-mor
source ~/.venvs/experiments-manifold-mor/bin/activate
```

### Installation of package via pip
Make sure that the virtual environment is active. Then change to the folder `hamiltonian-models` and run

```
pip install pip==23.3.1
pip install -e .
```

### Alternative installation
Alternatively, you may try to install newer versions of python and dependencies. You have to modify (a) the required python version and (b) the versions of dependencies in `pyproject.toml` before installing the package with pip.

## Running demos
In the demos folder, two scripts are provided, which showcase the implemented models.

```
python demos/models_lowdim.py
python demos/models_wave.py
```