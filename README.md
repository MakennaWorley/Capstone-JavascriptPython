# Probabilistic Ancestral Inference Research Project

A full-stack research framework for reconstructing missing ancestral genotypes from incomplete genetic data using probabilistic machine-learning models. The system uses `msprime` to simulate controlled, species-agnostic diploid populations with explicit multi-generational pedigrees, systematically masks genotype data to create realistic missingness, and benchmarks multiple inference architectures against the known ground truth.

### What This Project Does

In genetics research, it is common to have partial family data — grandparents or earlier ancestors may be unsequenced, deceased, or otherwise unavailable. Without that data, downstream analyses like disease risk prediction, inheritance tracing, and population history reconstruction are limited or impossible.

This project approaches that problem computationally. Rather than relying on a single model, it implements and compares **five distinct inference architectures** — spanning Bayesian statistics, hidden Markov models, deep learning, and graph neural networks — to determine which approaches best recover ground-truth genotypes when varying fractions of the family tree are missing.

The full pipeline is self-contained and reproducible:

1. **Simulate** — `msprime` generates realistic diploid populations with explicit multi-generational pedigrees using a Discrete Time Wright-Fisher model. Because the simulation is fully controlled, every ancestor's true genotype is known.
2. **Degrade** — Individuals are systematically masked at configurable rates, simulating the real-world condition of absent family members.
3. **Infer** — Each model is trained on the visible portion of the data and asked to predict the dosage (0, 1, or 2 copies of the alternate allele) for every masked individual at every genomic site.
4. **Evaluate** — Predictions are compared against the known ground truth using precision, recall, F1, ROC/PR curves, and calibration plots.
5. **Visualize** — A React dashboard presents dataset summaries, interactive family tree graphs, and per-model evaluation results in real time.

The goal is not simply to build a working imputation tool, but to rigorously characterize **when and why** each class of model succeeds or fails as data becomes increasingly sparse.

---

## Table of Contents

- [Project Goal](#project-goal)
- [Architecture Overview](#architecture-overview)
- [Technical Stack](#technical-stack)
- [Models](#models)
- [Evaluation Metrics](#evaluation-metrics)
- [Getting Started](#getting-started)
  - [Docker (Recommended)](#docker-recommended)
  - [NVIDIA GPU Support](#nvidia-gpu-support)
  - [Local Development (Without Docker)](#local-development-without-docker)
- [Project Structure](#project-structure)
- [API Reference](#api-reference)

---

## Project Goal

The core problem is **incomplete ancestry data**. Sequencing every ancestor is often impossible due to cost, sample degradation, or ethical constraints. This leaves gaps in family trees that limit the ability to reconstruct inheritance patterns, predict hereditary traits, or model population history.

This project evaluates whether probabilistic models can reconstruct missing ancestral genotypes, even when large portions of data are absent. It benchmarks the trade-offs between computational efficiency and inference precision across five model architectures.

**Key objectives:**

- Determine whether probabilistic models can reconstruct ancestral genotypes under controlled data degradation
- Compare model behavior under increasing uncertainty (Bayesian, HMM, DNN, GNN, and frequentist baselines)
- Identify the limits of inference when data becomes sparse, and characterize why reconstruction breaks down
- Provide a reproducible, containerized system for running these experiments

---

## Architecture Overview

The system follows a simulation-to-inference pipeline:

```
Simulate Population (msprime)
    |
    v
Generate Ground Truth Genotypes
    |
    v
Mask Individuals (controlled missingness)
    |
    v
Feature Engineering (k-hop relative aggregation)
    |
    v
Train / Evaluate Models (5 architectures)
    |
    v
Visualize Results (React Dashboard)
```

**Data flow details:**

1. **Simulation**: `msprime` generates multi-generational populations using a Discrete Time Wright-Fisher (DTWF) model with explicit pedigree tracking.
2. **Masking**: Entire individuals (columns) are masked at a configurable rate to simulate unobserved ancestors.
3. **Feature engineering**: For each masked individual, features are constructed from k-hop relatives (parents, grandparents, siblings, etc.) in the pedigree graph. Per-site features: `[mean_dosage_of_relatives, fraction_observed, count_relatives]`.
4. **Training**: A 3-phase pipeline (train, cross-validate + retrain on train+val, final test on held-out data).
5. **Visualization**: The React dashboard displays datasets, family trees, model metrics, and evaluation graphs.

---

## Technical Stack

| Layer | Technology |
|-------|-----------|
| **Data simulation** | `msprime`, `tskit` |
| **Backend API** | FastAPI, Uvicorn |
| **Bayesian inference** | PyMC, ArviZ |
| **HMM** | hmmlearn |
| **Deep learning** | PyTorch, PyTorch Geometric |
| **Scientific computing** | NumPy, pandas, SciPy, scikit-learn |
| **Visualization** | matplotlib, seaborn, Graphviz |
| **Frontend** | React 19, TypeScript, Material UI, RxJS |
| **Build / Lint** | Vite, Vitest, Biome (frontend); Ruff, pytest (backend) |
| **Deployment** | Docker Compose |

---

## Models

All five models implement a shared interface: `fit()`, `predict()`, `predict_proba()`, `predict_class()`, `save()`, `load()`.

| Model | Architecture | Key Properties |
|-------|-------------|----------------|
| **Bayesian Categorical** | Multinomial logistic regression with hierarchical priors (PyMC MCMC) | Interpretable uncertainty estimates; group-level intercepts per generation; optional JAX/GPU acceleration |
| **HMM Dosage** | Gaussian HMM (hmmlearn) | Treats each individual as a sequence of genetic sites; semi-supervised initialization from label statistics; 3 hidden states mapped to dosage classes |
| **DNN Dosage** | Fully connected neural network (PyTorch) | BatchNorm, dropout, optional residual connections; class-weighted loss; early stopping; CUDA/MPS support |
| **GNN Dosage** | Graph convolutional network (PyTorch Geometric) | Builds feature correlation graph; edges where correlation exceeds threshold; GraphConv layers with global mean pooling |
| **Multinomial Logistic Regression** | sklearn LogisticRegression | Frequentist baseline; balanced class weights; fastest to train |

---

## Evaluation Metrics

- **Reconstruction accuracy**: Precision, recall, and F1-scores (macro and weighted) against ground truth
- **ROC / PR curves**: Per-class (dosage 0, 1, 2) with AUC scores
- **Confusion matrices**: Heatmap visualization of predicted vs. true dosage classes
- **Model calibration**: Alignment of confidence intervals with actual recovery rates
- **Computational robustness**: Performance degradation across a spectrum of masking rates
- **Statistical significance**: Chi-square and likelihood-ratio tests

---

## Getting Started

### Docker (Recommended)

```bash
docker compose up --build
```

This starts two containers:

| Service | URL |
|---------|-----|
| FastAPI backend | `http://localhost:8000` |
| React frontend | `http://localhost:5173` |

The frontend proxies API requests to the backend automatically.

### NVIDIA GPU Support

If you have an NVIDIA GPU and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed:

```bash
docker compose --profile gpu up --build
```

This starts the GPU-accelerated backend (`backend_gpu`) instead of the CPU version. The same two URLs apply.

To build or run only the GPU backend:

```bash
docker compose build backend_gpu
docker compose up backend_gpu
```

### Rebuilding Containers

```bash
# Stop and remove containers
docker compose down

# Force rebuild (fresh dependencies)
docker compose build --no-cache

# Restart
docker compose up
```

### Local Development (Without Docker)

You can run the backend and frontend directly using the Makefile:

```bash
# Run backend only
make dev-back

# Run frontend only
make dev-front

# Run both concurrently
make dev
```

**Backend requirements**: Python 3.12, system-level Graphviz. See the [backend README](backend/README.md) for full conda environment setup.

**Frontend requirements**: Node.js 22+. Run `npm install` in the `frontend/` directory.

---

## Project Structure

```
.
├── docker-compose.yml          # Container orchestration (backend, frontend, GPU variant)
├── Makefile                    # Local dev shortcuts (dev-back, dev-front, dev)
├── pyproject.toml              # Ruff linter/formatter configuration
│
├── backend/
│   ├── Dockerfile.cpu          # Production CPU image (python:3.12-slim)
│   ├── Dockerfile.gpu          # NVIDIA GPU image (cuda:12.8.1 + conda)
│   ├── requirements.txt        # Production Python dependencies
│   ├── requirements_local.txt  # Development dependencies (testing, linting)
│   ├── gpu_setup.py            # GPU diagnostics and JAX/CUDA verification
│   └── app/
│       ├── main.py             # FastAPI application and API endpoints
│       ├── functions.py        # Shared utilities, response wrappers, file I/O
│       ├── data_generation.py  # Population simulation engine (msprime)
│       ├── data_preparation.py # Feature engineering from pedigree graphs
│       ├── model_main.py       # Model orchestration and training pipeline
│       ├── model_bayesian.py   # Bayesian categorical classifier (PyMC)
│       ├── model_hmm.py        # HMM dosage classifier (hmmlearn)
│       ├── model_dnn.py        # Deep neural network classifier (PyTorch)
│       ├── model_gnn.py        # Graph neural network classifier (PyG)
│       ├── model_multi_log_regression.py  # Multinomial logistic regression (sklearn)
│       ├── model_functions.py  # Shared model utilities (normalization, paths)
│       ├── model_graph_functions.py  # Evaluation graphing (ROC, PR, confusion)
│       ├── optimize_system.py  # System tuning for high-performance training
│       ├── datasets/           # Generated simulation data
│       ├── models/             # Trained model artifacts
│       ├── images/             # Evaluation plots
│       └── logs/               # Training logs
│
├── frontend/
│   ├── Dockerfile              # Node.js dev image (node:22-alpine)
│   ├── src/
│   │   ├── App.tsx             # Main app layout, state management
│   │   ├── components/         # UI components (dashboard, selectors, tree viz)
│   │   ├── hooks/              # RxJS polling hooks for datasets and models
│   │   └── services/           # API service layer (observable-based)
│   └── ...
│
└── streamlit_app/              # (Removed) Previously a Streamlit PoC dashboard
```

---

## API Reference

All endpoints return standardized JSON: `{"status": "success"|"error", "message": "...", "data": {...}}`.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/hello` | Health check |
| `POST` | `/api/create/data` | Create a new simulated dataset from parameters |
| `GET` | `/api/datasets/list` | List all available dataset names |
| `GET` | `/api/dataset/{name}/dashboard` | Fetch observed and truth genotype CSVs for display |
| `GET` | `/api/dataset/{name}/tree/{id}` | Get family tree subgraph (nodes, edges, genotypes) for an individual |
| `GET` | `/api/dataset/{name}/download` | Download all dataset files as a ZIP archive |
| `GET` | `/api/models/list` | List all trained models (name + type) |
| `POST` | `/api/models/test` | Test a trained model on a dataset; returns metrics, graphs, and per-prediction errors |
