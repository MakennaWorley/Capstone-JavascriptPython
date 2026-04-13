# Backend -- FastAPI + Inference Models

The backend is a FastAPI application that serves the REST API, runs population simulations via `msprime`, and trains/evaluates five probabilistic model architectures for ancestral genotype reconstruction.

---

## Table of Contents

- [Local Environment Setup](#local-environment-setup)
  - [Conda Environment](#conda-environment)
  - [PyTorch (DNN/GNN Models)](#pytorch-dnngnn-models)
  - [JAX (Bayesian GPU Acceleration)](#jax-bayesian-gpu-acceleration)
  - [System Dependencies](#system-dependencies)
- [Running the Backend](#running-the-backend)
- [Running Tests](#running-tests)
- [API Endpoints](#api-endpoints)
- [Data Generation](#data-generation)
  - [How data_generation.py Works](#how-data_generationpy-works)
  - [CLI Arguments](#cli-arguments)
  - [Example Commands](#example-commands)
  - [Simulation Presets](#simulation-presets)
  - [Output Files](#output-files)
- [Data Preparation](#data-preparation)
- [Model Training Pipeline](#model-training-pipeline)
  - [Three-Phase Training](#three-phase-training)
  - [Model Architectures](#model-architectures)
  - [Testing a Model on New Data](#testing-a-model-on-new-data)
- [Understanding the Data](#understanding-the-data)
- [File Reference](#file-reference)

---

## Local Environment Setup

### Conda Environment

Create and activate a Python 3.12 environment:

```bash
conda create -y -n capstone python=3.12
conda activate capstone
```

Install production dependencies:

```bash
pip install -r requirements.txt
```

For development (testing, linting, coverage, build tools):

```bash
pip install -r requirements_local.txt
```

### PyTorch (DNN/GNN Models)

PyTorch is not included in `requirements.txt` because the install command differs by platform. Install separately:

```bash
# CPU only (macOS / no GPU)
pip install torch torchvision torchaudio torch-geometric

# NVIDIA GPU (CUDA 12.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install torch-geometric
```

### JAX (Bayesian GPU Acceleration)

Optional. Improves PyMC sampling performance for the Bayesian model:

```bash
# CPU only
pip install jax jaxlib

# NVIDIA GPU
pip install "jax[cuda12]"
```

### System Dependencies

Graphviz is required at the system level for pedigree SVG generation:

```bash
# macOS
brew install graphviz

# Ubuntu / Debian
sudo apt-get install graphviz
```

---

## Running the Backend

From the repository root:

```bash
# Via Makefile
make dev-back

# Or directly
uvicorn backend.app.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

---

## Running Tests

```bash
cd backend/app
pytest
```

With coverage:

```bash
pytest --cov=. --cov-report=term-missing
```

---

## API Endpoints

All responses use the format `{"status": "success"|"error", "message": "...", "data": {...}}`.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/hello` | Health check |
| `POST` | `/api/create/data` | Create a simulated dataset from JSON parameters |
| `GET` | `/api/datasets/list` | List all dataset names |
| `GET` | `/api/dataset/{name}/dashboard` | Return observed + truth CSVs as text |
| `GET` | `/api/dataset/{name}/tree/{id}` | Return family tree subgraph for one individual |
| `GET` | `/api/dataset/{name}/download` | Download all dataset files as a ZIP |
| `GET` | `/api/models/list` | List trained models (name + type) |
| `POST` | `/api/models/test` | Test a model on a dataset; returns metrics, graphs, prediction errors |

---

## Data Generation

### How data_generation.py Works

The simulation engine uses `msprime` to generate multi-generational populations:

1. **Pedigree construction**: Builds an explicit multi-generation pedigree where every non-founder individual has two parents, creating a connected family forest.
2. **Tree sequence simulation**: Runs `msprime.sim_ancestry()` with the Discrete Time Wright-Fisher (DTWF) model and explicit pedigree tracking, then overlays mutations via `sim_mutations()`.
3. **Diploid conversion**: Converts haploid genotype matrices into diploid dosage values (0, 1, or 2) representing the count of derived alleles.
4. **Controlled masking**: Entire individuals (columns) are masked at the configured masking rate, setting all their genotypes to `NaN`. This simulates unobserved ancestors.
5. **Pedigree visualization**: Generates an SVG family tree using Graphviz, with dotted circles for masked individuals and solid circles for observed ones.
6. **Metadata export**: Saves every parameter and random seed to a JSON file for exact reproducibility.
7. **Retry logic**: If the mutation rate and seed produce fewer variants than `--min-variants`, the script increments the seed and retries automatically.

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--name` | (required) | Base name for output files |
| `--n-diploid-samples` | 250 | Number of diploid individuals |
| `--Ne` | 500 | Effective population size |
| `--ploidy` | 2 | Ploidy level |
| `--sequence-length` | 100 | Genome sequence length |
| `--mutation-rate` | 1e-8 | Mutation rate per base per generation |
| `--founder-recessive-chance` | 0.4 | Probability of recessive allele in founders |
| `--n_generations` | 5 | Number of generations to simulate |
| `--samples_per_generation` | 50 | Individuals sampled per generation |
| `--seed` | 42 | Random seed |
| `--masking-rate` | 0.20 | Fraction of individuals to mask |
| `--output-dir` | `datasets/` | Output directory |
| `--full-data` | False | Generate unmasked data only |
| `--meta-in` | None | Path to metadata JSON for exact replay |
| `--checks` | False | Run validation checks |

### Example Commands

Generate a dataset:

```bash
cd backend/app
python data_generation.py --name public --seed 21
```

Replay a previous run from its metadata (exact reproduction):

```bash
python data_generation.py --meta-in datasets/public.run_metadata.json
```

### Simulation Presets

When creating datasets through the API, these presets scale the simulation parameters:

| Preset | Samples | Sequence Length | Generations | Samples/Gen |
|--------|---------|----------------|-------------|-------------|
| `tiny` | Small | Short | Few | Few |
| `small` | Moderate | Moderate | Moderate | Moderate |
| `medium` | Moderate-large | Moderate-large | Moderate | Moderate |

The API enforces safety limits: max 1000 samples, max 1000 sequence length, max 10 generations.

### Output Files

Each dataset produces the following files (prefixed by name):

| File | Description |
|------|-------------|
| `*.truth_genotypes.csv` | Ground truth dosage matrix. Columns are individuals, rows are variant sites. Values: 0, 1, or 2. |
| `*.observed_genotypes.csv` | Copy of truth with masked individuals set to `NaN`. This is the input for models. |
| `*.pedigree.csv` | Maps individuals to parents and generations. Columns: `individual_id`, `time`, `parent_0_id`, `parent_1_id`, `num_nodes`. |
| `*.pedigree.svg` | SVG visualization of the family tree. |
| `*.run_metadata.json` | All parameters and seeds for exact replay. |
| `*.trees` | Tree sequence file (tskit format) containing the full genealogical history. |

---

## Data Preparation

`data_preparation.py` handles feature engineering for the models:

1. **Load data**: Reads truth genotypes, observed genotypes, and pedigree CSVs into DataFrames.
2. **Build adjacency graph**: Constructs an undirected parent-child adjacency list from the pedigree.
3. **Find families**: Identifies connected components (family clusters) via depth-first search.
4. **k-hop neighborhoods**: For each target individual, finds all relatives within k hops (default: 2) in the pedigree graph.
5. **Feature construction**: For each variant site of a target individual, computes three features from the k-hop relatives' observed genotypes:
   - Mean dosage of observed relatives at that site
   - Fraction of relatives that are observed (not masked)
   - Count of relatives in the neighborhood
6. **Train/val/test split**: Splits by family groups (default 70/15/15) to prevent data leakage between related individuals.
7. **Class resampling**: Oversamples minority dosage classes in training data for balanced learning.

---

## Model Training Pipeline

### Three-Phase Training

`model_main.py` orchestrates a three-phase pipeline:

1. **Phase 1 -- Train**: Train on the training split with class-balanced resampling.
2. **Phase 2 -- Validate**: K-fold cross-validation (k=5) on the validation split, then retrain on combined train+val data.
3. **Phase 3 -- Test**: Final evaluation on the held-out test split. Saves ROC/PR curves, confusion matrices, and metrics.

### Model Architectures

**Bayesian Categorical Dosage Classifier** (`model_bayesian.py`): Multinomial logistic regression with softmax over 3 classes using PyMC. Hierarchical priors with group-level intercepts per generation. Uses MCMC sampling. Configurable draws, tune, chains, target acceptance rate. Optional JAX GPU backend.

**HMM Dosage Classifier** (`model_hmm.py`): Gaussian HMM with 3 hidden states corresponding to dosage classes. Treats each individual as a sequence where timesteps are genetic sites. Semi-supervised initialization from label statistics. State-to-dosage alignment via greedy count-matrix matching.

**DNN Dosage Classifier** (`model_dnn.py`): Fully connected neural network with configurable hidden dimensions (default 256, 128, 64). BatchNorm, ReLU, dropout, optional residual connections. Class-weighted cross-entropy loss. Early stopping on validation loss. Supports CUDA and Apple Metal (MPS).

**GNN Dosage Classifier** (`model_gnn.py`): Graph convolutional network using PyTorch Geometric. Constructs a feature correlation graph where edges connect variants with correlation above a threshold. Multiple GraphConv layers with global mean pooling. Falls back to k-NN graph if no correlated edges exist.

**Multinomial Logistic Regression** (`model_multi_log_regression.py`): Standard sklearn LogisticRegression with balanced class weights and StandardScaler preprocessing. Serves as the frequentist baseline. Fastest model to train.

### Testing a Model on New Data

`test_on_new_data()` in `model_main.py` loads a pre-trained model and applies it to a different dataset. It validates that feature dimensions match, runs prediction, computes metrics, generates evaluation graphs, and caches results in `applied_models.csv`. This is what the `/api/models/test` endpoint calls.

---

## Understanding the Data

### Column Reference

| Column | Description |
|--------|-------------|
| `index` / `site_index` | Zero-indexed identifier for each variant site |
| `position` | Genomic position of the variant |
| `i_0000` through `i_NNNN` | Each column is a diploid individual |
| `time` | (pedigree) Generation when the individual lived. 0 is the most recent. |
| `parent_0_id`, `parent_1_id` | (pedigree) Parent IDs. `-1` means the parent is outside the simulated range. |

### Dosage Values

The genotype columns contain diploid dosage values representing the count of the derived allele at each site:

| Value | Meaning |
|-------|---------|
| 0 | Homozygous ancestral (neither chromosome carries the mutation) |
| 1 | Heterozygous (one chromosome carries the mutation) |
| 2 | Homozygous derived (both chromosomes carry the mutation) |
| NaN | Missing / masked (in observed genotypes only) |

---

## File Reference

| File | Purpose |
|------|---------|
| `main.py` | FastAPI application, API endpoint definitions |
| `functions.py` | Shared utilities: response wrappers, file I/O, pedigree graph traversal |
| `data_generation.py` | Population simulation engine (msprime, DTWF, masking) |
| `data_preparation.py` | Feature engineering from pedigree k-hop neighborhoods |
| `model_main.py` | Training orchestration, 3-phase pipeline, system optimization |
| `model_bayesian.py` | Bayesian multinomial classifier (PyMC, MCMC) |
| `model_hmm.py` | Hidden Markov Model classifier (hmmlearn) |
| `model_dnn.py` | Deep neural network classifier (PyTorch) |
| `model_gnn.py` | Graph neural network classifier (PyTorch Geometric) |
| `model_multi_log_regression.py` | Multinomial logistic regression baseline (sklearn) |
| `model_functions.py` | Shared model utilities: normalization, path management, metadata I/O |
| `model_graph_functions.py` | Evaluation graphing: ROC, PR, confusion matrix, regression diagnostics |
| `optimize_system.py` | System-level tuning for high-performance training (env vars, threading) |
| `debug_covars.py` | Debug script for HMM covariance serialization |
| `gpu_setup.py` | GPU diagnostics: verifies NVIDIA drivers, CUDA, JAX, PyMC GPU support |
| `requirements.txt` | Production Python dependencies (Docker) |
| `requirements_local.txt` | Development dependencies (extends requirements.txt with testing/linting) |
| `Dockerfile.cpu` | CPU Docker image (python:3.12-slim) |
| `Dockerfile.gpu` | GPU Docker image (nvidia/cuda + conda) |
