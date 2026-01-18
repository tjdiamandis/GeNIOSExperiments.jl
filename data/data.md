# Dataset Dependencies

This document describes all the datasets required to run the experiments in GeNIOSExperiments.jl and how to obtain them.

## Required Datasets

### 1. YearPredictionMSD Dataset

**File**: `YearPredictionMSD.txt`  
**Used by**: Elastic net experiments, constrained least squares experiments, logistic regression experiments  
**Location**: `experiments/data/YearPredictionMSD.txt`

**Description**:
- Prediction of song release year from audio features
- 515,345 instances with 90 features (12 timbre average + 78 timbre covariance)
- Songs range from 1922 to 2011, mostly western commercial tracks
- Features extracted from Echo Nest API timbre data

**How to obtain**:
1. Download from UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/203/yearpredictionmsd
2. Place the downloaded `YearPredictionMSD.txt` file in `experiments/data/`

### 2. Real-sim Dataset (OpenML 1578)

**File**: `real-sim.jld2` (generated automatically)  
**Used by**: Elastic net experiments, logistic regression experiments  
**Location**: `experiments/data/real-sim.jld2`

**Description**:
- Binary classification dataset from LIBSVM repository
- Sparse dataset preprocessed by Vikas Sindhwani for SVMlin project
- Originally from A. McCallum

**How to obtain**:
The dataset is automatically downloaded using OpenML when you first run experiments:
```julia
using OpenML
dataset = OpenML.load(1578)  # Downloads real-sim dataset
```

Alternatively, you can download manually:
1. Visit: https://www.openml.org/d/1578
2. The experiments will automatically cache it as `real-sim.jld2`


## Setup Instructions

1. **Create data directory**:
   ```bash
   mkdir -p experiments/data
   ```

2. **Download YearPredictionMSD manually**:
   - Download `YearPredictionMSD.txt` from UCI repository
   - Place in `experiments/data/YearPredictionMSD.txt`

3. **OpenML datasets** (automatic):
   - The OpenML datasets (real-sim, news20) will be downloaded automatically on first use
   - Ensure you have internet connection when running experiments for the first time
   - Set `HAVE_DATA_SPARSE = false` in experiment files if you want to force re-download

## Dataset Usage Flags

In the experiment files, you'll find these flags that control dataset loading:

- `HAVE_DATA_SPARSE = true`: Set to `false` to force re-download of OpenML datasets
- `HAVE_DATA_SPARSE2 = true`: Controls second sparse dataset (news20) availability

## Storage Requirements

- `YearPredictionMSD.txt`: ~450 MB
- `real-sim.jld2`: ~60 MB
