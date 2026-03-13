# Notebook

## Environment setup

There is a complete installation instruction in the [README.md](../../README.md) of the SDB GUI repository, but I will summarize it here for your convenience.

### 1. Install package management tool

Install either `conda` or `pixi`.

### 2. Source code download

You can download the source code directly from the [sdb_gui repository](https://github.com/rifqiharrys/sdb_gui) or clone the repository using git, then move into the source code directory.

```bash
    git clone https://github.com/rifqiharrys/sdb_gui.git
    cd sdb_gui
```

### 3. Install packages

Install the required packages using `conda` or `pixi`. If you use `conda`, install the packages using the `environment.yaml`, then install `jupyterlab` afterwards using the following command.

```bash
    conda env create -f environment.yaml
    conda install -n sdb-gui jupyterlab
```

If you use `pixi`, there is a dedicated environment called `notebook` written in `pixi.toml` file, so you can install the packages using the following command.

```bash
    pixi install -e notebook
```

### 4. Run the notebook

After installing the packages, activate the environment and you can run the notebook by running the following command.

```bash
    jupyter lab
```

Or you could run the notebook using VS Code, but make sure to select the correct environment that you have installed the packages in.
