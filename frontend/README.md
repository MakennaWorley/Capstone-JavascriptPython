# Frontend -- React + TypeScript Dashboard

An interactive visualization dashboard built with React 19, TypeScript, and Material UI. Connects to the FastAPI backend to display datasets, family trees, trained models, and evaluation metrics.

---

## Table of Contents

- [Local Development](#local-development)
- [Scripts](#scripts)
- [Architecture](#architecture)
  - [App Structure](#app-structure)
  - [Components](#components)
  - [Hooks](#hooks)
  - [Services](#services)
- [Configuration](#configuration)
  - [Environment Variables](#environment-variables)
  - [Vite Config](#vite-config)
  - [Biome (Linting / Formatting)](#biome-linting--formatting)
- [Testing](#testing)
- [Dependencies](#dependencies)

---

## Local Development

Requires Node.js 22+.

```bash
# Install dependencies
npm install

# Start dev server
npm run dev
```

The dev server runs at `http://localhost:5173` and proxies `/fastapi` requests to the backend at `http://localhost:8000` by default.

When running via Docker, the proxy target is set through the `VITE_PROXY_HOST` environment variable to point to the `fastapi` container.

---

## Scripts

| Script | Command | Description |
|--------|---------|-------------|
| `dev` | `npm run dev` | Start Vite dev server with HMR |
| `build` | `npm run build` | Production build |
| `preview` | `npm run preview` | Preview production build locally |
| `format` | `npm run format` | Auto-format all files with Biome |
| `format:check` | `npm run format:check` | Check formatting without modifying |
| `lint` | `npm run lint` | Run Biome lint rules |
| `check` | `npm run check` | Run all Biome checks (lint + format) |
| `test` | `npm run test` | Run Vitest in watch mode |
| `test:run` | `npm run test:run` | Run tests once |
| `test:cov` | `npm run test:cov` | Run tests with V8 coverage report |

---

## Architecture

### App Structure

`App.tsx` is the top-level component. It manages:

- **Theme**: Material UI dark/light mode, toggleable and defaults to system preference
- **State**: Selected dataset, selected model, test results, debug mode, dark mode, create-dataset modal
- **Polling**: RxJS-based observables that poll the backend for datasets and models on a 5-second interval
- **Layout**: Collapsible sidebar + main content area with accordion panels for Dataset, Model, and Test sections

### Components

| Component | File | Description |
|-----------|------|-------------|
| `DatasetDisplayDashboard` | `DatasetDisplayDashboard.tsx` | Fetches and displays observed/truth genotype CSVs. Includes paginated CSV preview table, download button, and embedded family tree visualization. |
| `DatasetModelCreationForm` | `DatasetModelCreationForm.tsx` | Form for creating new simulated datasets. Basic mode accepts a name; advanced mode exposes `sequence_length`, `n_generations`, and `samples_per_generation`. Enforces the same safety limits as the backend (max 1000 samples, 1000 sequence length, 10 generations). |
| `DatasetSelector` | `DatasetSelector.tsx` | Dropdown for choosing from available datasets. Sorted alphabetically. |
| `FamilyTreeVisualization` | `FamilyTreeVisualization.tsx` | SVG-based interactive family tree. Nodes are positioned by generation on the Y-axis, with edges connecting parents to children. Hover reveals node details (individual ID, generation, observed status). Custom layout algorithm handles parent/child direction detection. |
| `LoadingProgress` | `LoadingProgress.tsx` | Animated loading indicator with pulsing dots and a progress bar. Uses ARIA attributes for accessibility (`role="status"`, `aria-live="polite"`). |
| `ModelDashboard` | `ModelDashboard.tsx` | Information panel for the selected model type. Contains descriptions, strengths, and use cases for each of the five model architectures. Has a toggle between "simple" and "technical" description modes. |
| `ModelSelector` | `ModelSelector.tsx` | Dropdown for choosing a trained model. Displays human-readable type names (e.g., "Bayesian Inference", "Hidden Markov Model"). |
| `ModelStats` | `ModelStats.tsx` | Displays test results: metrics table (accuracy, balanced accuracy, AUC, F1), human-readable and technical descriptions for each metric, base64-encoded ROC/PR and confusion matrix images, and a paginated prediction error table. |
| `ModelTrainer` | `ModelTrainer.tsx` | "Run Test" button that POSTs to `/api/models/test` with the selected dataset and model. Shows a loading indicator during execution. |
| `Sidebar` | `Sidebar.tsx` | Persistent left drawer, collapsible between 80px and 280px. Contains: Create Dataset button, dark/light mode toggle, debug mode toggle. |

### Hooks

**`useApiPolling.ts`** exports two hooks:

- `useDatasetsPoll(apiBase, apiKey, intervalMs)`: Subscribes to an RxJS observable that polls `GET /api/datasets/list` at the given interval. Returns `{ datasets, error, isLoading, refresh }`.
- `useModelsPoll(apiBase, apiKey, intervalMs)`: Same pattern for `GET /api/models/list`. Returns `{ models, error, isLoading }`.

Both hooks clean up their subscriptions on component unmount.

### Services

**`apiService.ts`**: Class-based API service using RxJS observables.

- `fetchDatasetsOnce()` / `fetchModelsOnce()`: Raw `fetch()` calls with `x-api-key` header support.
- `createDatasetsPoll(intervalMs)` / `createModelsPoll(intervalMs)`: Return `Observable<T>` using `interval().pipe(startWith, switchMap, catchError)` for reactive polling.

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VITE_API_BASE` | `/fastapi` | URL prefix for backend API requests (used as Vite proxy path) |
| `VITE_PROXY_HOST` | `http://localhost:8000` | Backend host for Vite dev server proxy target |
| `VITE_X_API_KEY` | (none) | Optional API key sent in `x-api-key` header |

### Vite Config

`vite.config.ts` configures:

- React plugin (`@vitejs/plugin-react`)
- Dev server on port 5173
- Proxy: requests to `VITE_API_BASE` are forwarded to `VITE_PROXY_HOST` with path rewriting

### Biome (Linting / Formatting)

`biome.json` configures Biome 2.4.1:

- **Indent**: Tabs, width 4
- **Line endings**: LF
- **Line width**: 150
- **Quotes**: Single
- **Semicolons**: Always
- **Trailing commas**: None
- **Imports**: Auto-organized
- **Lint rules**: Recommended set enabled

---

## Testing

Tests use [Vitest](https://vitest.dev/) with [Testing Library](https://testing-library.com/docs/react-testing-library/intro/) and jsdom.

```bash
# Watch mode
npm run test

# Single run
npm run test:run

# With coverage
npm run test:cov
```

Coverage is generated using the V8 provider and output to the `coverage/` directory. Coverage excludes test files, `main.tsx`, and env declaration files.

---

## Dependencies

### Runtime

| Package | Purpose |
|---------|---------|
| `react` / `react-dom` | UI framework |
| `@mui/material` / `@mui/icons-material` | Component library and icons |
| `@emotion/react` / `@emotion/styled` | CSS-in-JS (required by MUI) |
| `papaparse` | CSV parsing for dataset display |
| `rxjs` | Reactive polling for datasets and models |

### Development

| Package | Purpose |
|---------|---------|
| `typescript` | Type checking |
| `vite` | Build tool and dev server |
| `@vitejs/plugin-react` | React support for Vite |
| `vitest` | Test runner |
| `@vitest/coverage-v8` | Code coverage |
| `@testing-library/react` | Component testing utilities |
| `@testing-library/user-event` | User interaction simulation |
| `@testing-library/jest-dom` | Custom DOM matchers |
| `jsdom` | Browser environment for tests |
| `@biomejs/biome` | Linting and formatting |
| `sass` / `sass-embedded` | SCSS support |
