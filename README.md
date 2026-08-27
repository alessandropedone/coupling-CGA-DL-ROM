
# Introduction

## Setup

Clone the repository with
```bash
git clone git@github.com:alessandropedone/deeponet-for-mems.git
```

> Conda (or Mamba) is required.

Create the project environment by running this command (or the equivalent Conda command):
```bash
mamba env create -f environment.yml
```

## Structure of the repository
```bash
deeponet-for-mems
├── README.md
├── build_docs.sh
├── docs
├── environment.yml
├── geometries
├── models
├── report.pdf
├── src
│   ├── data
│   ├── multi_physics
│   └── surrogate
└── test
```

## Report
For a detailed explanation of all functionalities and the theoretical background, please consult `report.pdf` in the main directory.

## Documentation
The documentation is present in the `docs` folder in HTML format, and online [here](https://alessandropedone.github.io/deeponet-for-mems/).
The deeper version with private functions is compiled, 
but if you want the more compact version with only the public functions, 
you can just run the following command, and select the right option, to generate a new version of the documentation.
```bash
./build_docs.sh
```

## Numerical results
You can find the instruction on how to reproduce them in [`test/run_test_cases.md`](https://github.com/alessandropedone/deeponet-for-mems/blob/main/test/run_test_cases.md), or equivalently in the documentation.

