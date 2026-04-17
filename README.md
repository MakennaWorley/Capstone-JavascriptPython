## 🧬 Probabilistic Ancestral Inference Research Project

This repository contains a research framework designed to reconstruct latent states within high-dimensional, hierarchical stochastic datasets. While the project utilizes biological rules for data generation, its primary purpose is benchmarking the robustness of various probabilistic machine-learning architectures under controlled data degradation.

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

### 🌟 Project Goal

The core mission is to evaluate the technical trade-offs between computational efficiency and inference precision. Key objectives include:

#### 🛠️ Key Components

* **Hierarchical Modeling:** Designing a system to model latent dependencies within stochastic data.
* **Multi-Model Benchmarking:** Implementing and comparing Bayesian Inference, Hidden Markov Models (HMM), and Graph Neural Networks (GNN)..
* **Quantitative Validation:** Using "ground-truth" data generators to measure recovery performance against systematic masking.
* **Visualization:** A React dashboard to visualize the data and models.

---

### 🛠 Technical Stack & Reproducibility

The project emphasizes engineering rigor and reproducibility through a modular architecture:

* **Data Engine:** Powered by `msprime` for high-fidelity simulation of multi-generational datasets.
* **Meta-Replay System:** Uses JSON-based metadata and specific random seeds to ensure exact dataset reconstruction for benchmarking.
* **Inference Frameworks:**
    * **Bayesian:** Developed via `PyMC` and `PyTensor`
    * **HMM:** Implemented using `pomegranate` or `hmmlearn`
    * **GNN:** Built with PyTorch `Geometric` for multi-way dependency modeling
* **Application Layer:** A `FastAPI` backend serving a custom `React` dashboard for interactive visualization of calibration and uncertainty.
* **Deployment:** Fully containerized with `Docker` for cross-platform portability.

---

### 📊 Evaluation Metrics

Models are benchmarked using both quantitative and qualitative dimensions:


* **Reconstruction Accuracy:** Precision, recall, and F1-scores compared against the known "truth".
* **Model Calibration:** Aligning reported confidence intervals with actual recovery rates.
* **Computational Robustness:** Measuring the "break point" of each architecture across a spectrum of masking rates.
* **Statistical Significance:** Validation via chi-square and likelihood-ratio tests.

---

### 🚀 Getting Started

The entire system will be containerized with Docker to ensure reproducibility and ease of deployment.

#### How to Run Locally (CPU default - Mac Friendly)

```bash
docker compose up --build
```

Once that finishes, you should see three containers running:
- FastAPI: `http://localhost:8000`
- React: `http://localhost:5173`
- Streamlit: `http://localhost:8501`

#### How to Run Locally (NVIDIA GPU)

The repository defaults to the CPU backend. If you have an NVIDIA GPU and the NVIDIA Container Toolkit installed, start the optional GPU backend using the `gpu` profile:

```bash
docker compose --profile gpu up --build
```

This enables the GPU-backed `backend_gpu` service (the CPU backend remains the default). After startup you should see the same three apps:
- FastAPI: `http://localhost:8000`
- React: `http://localhost:5173`
- Streamlit: `http://localhost:8501`

If you prefer to build/run only the GPU backend service directly:

```bash
docker compose build backend_gpu
docker compose up backend_gpu
```

Notes:
- The GPU service requires a host with NVIDIA drivers and the NVIDIA Container Toolkit.
- Use the `--profile gpu` option only on machines with a supported GPU.

#### How to Rebuild Containers (If you mess them up)
To ensure a clean environment, you can stop, remove, and rebuild your Docker containers:

1. Stop and Remove Running Containers:
```bash
docker compose down
```

2. Force Rebuild Images (Pulls fresh dependencies):

```bash
docker compose build --no-cache
```

3. Restart the Service:
```bash
docker compose up
```

#### How to run the .venv python environment

Bash: run `source .venv/bin/activate` to activate the python environment
Windows: run `.\.venv\Scripts\Activate.ps1`