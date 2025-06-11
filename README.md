# GeNIOSExperiments

# GeNIOSExperiments

This repository contains the experimental code used in the paper for the [GeNIOS.jl](https://github.com/tjdiamandis/GeNIOS.jl) solver. GeNIOS is a first-order solver for convex optimization problems.

## Overview

The experiments in this repository demonstrate GeNIOS's performance on various optimization problems, including:

- Portfolio optimization
- Constrained least squares
- Elastic net regression
- Logistic regression

## Repository Structure

- `experiments/`: Contains the main experiment scripts
  - `5-portfolio.jl`: Portfolio optimization experiments
  - `utils.jl`: Utility functions for data loading and problem construction
- `data/`: Directory for storing experiment data
- `saved/`: Directory for saving experiment results
- `figures/`: Directory for storing generated figures

## Usage

To run the experiments:

1. Clone this repository
2. Install the required dependencies
3. Run the desired experiment script, e.g.:
   ```julia
   julia experiments/5-portfolio.jl
   ```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citing
Please cite the associated paper:
```
@article{diamandis2023genios,
  title={GeNIOS: an (almost) second-order operator-splitting solver for large-scale convex optimization},
  author={Diamandis, Theo and Frangella, Zachary and Zhao, Shipu and Stellato, Bartolomeo and Udell, Madeleine},
  journal={arXiv preprint arXiv:2310.08333},
  year={2023}
}
```
