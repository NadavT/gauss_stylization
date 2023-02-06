# Gauss Stylization

This is a python implementation for the paper "Gauss Stylization: Interactive Artistic Mesh Modeling
based on Preferred Surface Normals" by M. Kohlbrenner, U. Finnendahl, T. Djuren, M. Alexa
See the [project page](http://www.martin-kohlbrenner.de/gauss_stylizationhttps://cybertron.cg.tu-berlin.de/projects/gaussStylization/) for more information.

## Running the code

### Setup

We used conda to manage the python environment.

You can install conda from [here](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html) (We recommend using miniconda as it is the lighter way to get the conda package manager).

To install the required packages, run

```sh
conda env create -f environment.yml
```

### Running the code

In a terminal, activate the conda environment by running

```sh
conda activate gauss_stylization
```

And then on the same terminal run

```sh
python app.py
```

Run help to see the available options

```sh
python app.py --help
```

```
usage: gauss_stylization [-h] [--model MODEL] [--sigma SIGMA] [--mu MU] [--lambda_value LAMBDA_VALUE]
                         [--caxiscontrib CAXISCONTRIB] [--admm_iterations ADMM_ITERATIONS]

options:
  -h, --help            show this help message and exit
  --model MODEL         model to stylize
  --sigma SIGMA         sigma value for function
  --mu MU               mu parameter
  --lambda_value LAMBDA_VALUE
                        lambda parameter
  --caxiscontrib CAXISCONTRIB
                        Axis contribution in semi-discrete normals (discrete normals contribution when using semi-
                        discrete normals)
  --admm_iterations ADMM_ITERATIONS
                        admm iterations to do per gauss stylization update
```
