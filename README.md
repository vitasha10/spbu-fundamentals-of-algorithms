# spbu-fundamentals-of-algorithms
Materials for the practicum for "Fundamentals of Algorithms" course at SpbU

## Getting started

Set up your python environment. It is recommended to use [miniconda](https://docs.anaconda.com/miniconda) and python 3.10 and higher. You will need to install the latest versions of numpy, matplotlib and networkx which can be done via running
```bash
$ cd /path/to/your/repo/clone
$ pip install -r requirements.txt
```

While working with this repo, you will need to be able to import functions from it which means it either should be added to PYTHONPATH or somehow installed to site-packages. Below we list several ways how this can be done.

### Terminal (any environment)

Run this command in the terminal and work within the same session:
```bash
export PYTHONPATH=/path/to/your/repo/clone:$PYTHONPATH
```
This command must be run again if a new session is created. To avoid this tedious move, just copy it to the config file of your terminal (e.g., `.bash_profile` or `.bash_rc`).

### Terminal (conda)

Run this command in the terminal with the conda environment being activated:
```bash
conda develop /path/to/your/repo/clone
```

### VSCode

Go to `Run and Debug` in the left panel, create a new launch file, select `Python File` and add the following field:
```yaml
"env": {
    "PYTHONPATH": "${workspaceFolder}${pathSeparator}${env:PYTHONPATH}"
}
```

### PyCharm

TODO

## Practicum 1

Изучение `python`, `numpy` и  `matplotlib`, необходимых для дальнейшей работы. Предполагается, что студент имеет базовые знания python.

План:
1. Выполнить `intro_to_numpy_and_matplotlib.ipynb`

