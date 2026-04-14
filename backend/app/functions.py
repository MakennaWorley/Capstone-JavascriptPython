from __future__ import annotations

import base64
import csv
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

import pandas as pd
from dotenv import load_dotenv
from fastapi.responses import JSONResponse

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
load_dotenv(dotenv_path=ROOT_DIR / '.env')

DATASETS_DIR = (BASE_DIR / os.getenv('DATASETS_DIR')).resolve()
PROTECTED_DATASETS_DIR = (BASE_DIR / os.getenv('PROTECTED_DATASETS_DIR')).resolve()
MODELS_DIR = (BASE_DIR / os.getenv('MODELS_DIR')).resolve()
IMAGES_DIR = (BASE_DIR / os.getenv('IMAGES_DIR')).resolve()
LOGS_DIR = (BASE_DIR / os.getenv('LOGS_DIR')).resolve()

# -----------------------------
# API
# -----------------------------


def api_success(message: str, data: Dict[str, Any], status_code: int = 200) -> JSONResponse:
	return JSONResponse(status_code=status_code, content={'status': 'success', 'message': message, 'data': data})


def api_error(message: str, status_code: int, code: str = 'error') -> JSONResponse:
	return JSONResponse(status_code=status_code, content={'status': 'error', 'code': code, 'message': message})


# -----------------------------
# Classes
# -----------------------------


class DashboardFilesMissing(FileNotFoundError):
	def __init__(self, dataset_name: str, missing: list[str]):
		super().__init__(f"Missing required files for dataset '{dataset_name}': {', '.join(missing)}")
		self.dataset_name = dataset_name
		self.missing = missing


# -----------------------------
# Helpers
# -----------------------------


def _paths_for_dataset(name: str, datasets_dir: Path = DATASETS_DIR) -> Dict[str, Path]:
	"""
	Mirrors data_generation.build_paths(cfg), but from just the dataset name/prefix.
	Returns canonical paths for ALL files we may want to return.
	"""
	base = datasets_dir / name
	base_str = str(base)
	return {
		'trees': Path(base_str + '.trees'),
		'truth_csv': Path(base_str + '.truth_genotypes.csv'),
		'observed_csv': Path(base_str + '.observed_genotypes.csv'),
		'pedigree_csv': Path(base_str + '.pedigree.csv'),
		'meta_json': Path(base_str + '.run_metadata.json'),
	}


# -----------------------------
# Functions
# -----------------------------


def col_name(individual_id: int) -> str:
	if individual_id < 0:
		raise ValueError('individual_id must be >= 0')
	return f'i_{individual_id:04d}'


def get_dataset_names() -> List[str]:
	"""
	Reads dataset names from datasets.csv.
	De-duplicates while preserving order.
	"""
	seen: Set[str] = set()
	names: List[str] = []

	file_path = DATASETS_DIR / 'datasets.csv'

	if not file_path.exists():
		return []

	with open(file_path, newline='', encoding='utf-8') as f:
		reader = csv.DictReader(f)
		for row in reader:
			name = row.get('dataset_name', '').strip()
			if name and name not in seen:
				seen.add(name)
				names.append(name)

	return names


def delete_old_datasets(datasets_dir: Path = DATASETS_DIR, max_age_days: int = 1) -> int:
	"""
	Delete all dataset files older than max_age_days and remove their entries
	from datasets.csv. Returns the number of datasets deleted.
	"""
	csv_path = datasets_dir / 'datasets.csv'
	if not csv_path.exists():
		return 0

	cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)

	to_delete: List[str] = []
	to_keep: List[Dict[str, str]] = []

	with open(csv_path, newline='', encoding='utf-8') as f:
		reader = csv.DictReader(f)
		for row in reader:
			try:
				date_str = row.get('date_created', '').rstrip('Z')
				date_created = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
				if date_created < cutoff and row.get('creator') == 'frontend':
					to_delete.append(row['dataset_name'])
				else:
					to_keep.append(row)
			except (ValueError, KeyError):
				to_keep.append(row)

	if not to_delete:
		return 0

	for dataset_name in to_delete:
		base = str(datasets_dir / dataset_name)
		for suffix in ['.trees', '.truth_genotypes.csv', '.observed_genotypes.csv', '.pedigree.csv', '.run_metadata.json', '.pedigree.svg']:
			path = Path(base + suffix)
			if path.exists():
				path.unlink()

	with open(csv_path, 'w', newline='', encoding='utf-8') as f:
		writer = csv.DictWriter(f, fieldnames=['dataset_name', 'creator', 'date_created'])
		writer.writeheader()
		writer.writerows(to_keep)

	return len(to_delete)


def delete_old_logs(logs_dir: Path = LOGS_DIR, max_age_days: int = 1) -> int:
	"""
	Delete all log files older than max_age_days and remove their entries
	from applied_models.csv. Returns the number of log entries deleted.
	"""
	csv_path = logs_dir / 'applied_models.csv'
	if not csv_path.exists():
		return 0

	cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)

	to_keep: List[Dict[str, str]] = []
	deleted_count = 0

	with open(csv_path, newline='', encoding='utf-8') as f:
		reader = csv.DictReader(f)
		fieldnames = reader.fieldnames or []
		for row in reader:
			try:
				date_str = row.get('applied_date', '').rstrip('Z')
				applied_date = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
				if applied_date < cutoff:
					for col in ('log_file', 'graph_test', 'graph_cm'):
						file_path = row.get(col, '').strip()
						if file_path:
							p = Path(file_path)
							if p.exists():
								p.unlink()
					deleted_count += 1
				else:
					to_keep.append(row)
			except (ValueError, KeyError):
				to_keep.append(row)

	if deleted_count == 0:
		return 0

	with open(csv_path, 'w', newline='', encoding='utf-8') as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(to_keep)

	return deleted_count


def get_model_list(models_dir: Path = MODELS_DIR) -> List[Dict[str, str]]:
	"""
	Reads trained models from models.csv in the models directory.
	Returns a list of dictionaries with 'model_name' and 'model_type' keys.

	Example return:
		[
			{'model_name': 'testing.training', 'model_type': 'bayes_softmax3'},
			{'model_name': 'testing.training', 'model_type': 'multi_log_regression'},
		]
	"""
	csv_path = models_dir / 'models.csv'

	if not csv_path.exists():
		raise FileNotFoundError(f'models.csv not found at {csv_path}')

	models: List[Dict[str, str]] = []

	with open(csv_path, 'r', newline='', encoding='utf-8') as f:
		reader = csv.DictReader(f)
		for row in reader:
			models.append({'model_name': row['model_name'], 'model_type': row['model_type']})

	return models


def get_file(path: Path, *, mode: Literal['text', 'base64'] = 'text', encoding: str = 'utf-8') -> Dict[str, Any]:
	"""
	Read ONE file from disk.

	- mode='text'   -> returns {"name": ..., "text": ...}
	- mode='base64' -> returns {"name": ..., "base64": ..., "byte_length": ...}
	"""
	if not path.exists():
		raise FileNotFoundError(str(path))

	if mode == 'text':
		return {'name': path.name, 'text': path.read_text(encoding=encoding)}

	# mode == 'base64'
	b = path.read_bytes()
	return {'name': path.name, 'base64': base64.b64encode(b).decode('ascii'), 'byte_length': len(b)}


def get_dataset_dashboard_files(dataset_name: str, datasets_dir: Path = DATASETS_DIR) -> Dict[str, Any]:
	"""
	Build the dashboard response data for one dataset.
	"""
	truth_path = datasets_dir / f'{dataset_name}.truth_genotypes.csv'
	observed_path = datasets_dir / f'{dataset_name}.observed_genotypes.csv'

	missing: list[str] = []
	if not truth_path.exists():
		missing.append(truth_path.name)
	if not observed_path.exists():
		missing.append(observed_path.name)

	if missing:
		raise DashboardFilesMissing(dataset_name=dataset_name, missing=missing)

	# Use get_file multiple times (single-responsibility)
	truth = get_file(truth_path, mode='text')
	observed = get_file(observed_path, mode='text')

	return {'dataset': dataset_name, 'observed_genotypes_csv': observed['text'], 'truth_genotypes_csv': truth['text']}


def get_all_dataset_files(dataset_name: str, datasets_dir: Path = DATASETS_DIR) -> Dict[str, Any]:
	"""
	Return ALL dataset files for a dataset prefix.
	Uses _paths_for_dataset + calls get_file() multiple times (one per file).
	"""
	paths = _paths_for_dataset(dataset_name, datasets_dir=datasets_dir)

	missing = [p.name for p in paths.values() if not p.exists()]
	if missing:
		raise DashboardFilesMissing(dataset_name=dataset_name, missing=missing)

	trees = get_file(paths['trees'], mode='base64')
	truth = get_file(paths['truth_csv'], mode='text')
	observed = get_file(paths['observed_csv'], mode='text')
	pedigree = get_file(paths['pedigree_csv'], mode='text')
	meta = get_file(paths['meta_json'], mode='text')

	return {
		'dataset': dataset_name,
		# CSVs / JSON as text
		'truth_genotypes_csv': truth['text'],
		'observed_genotypes_csv': observed['text'],
		'pedigree_csv': pedigree['text'],
		'run_metadata_json': meta['text'],
		# trees as base64
		'trees_name': trees['name'],
		'trees_base64': trees['base64'],
		'trees_byte_length': trees['byte_length'],
	}


def get_genetic_data_from_id(
	dataset_name: str, individual_id: int, *, observed_df: Optional[pd.DataFrame] = None, datasets_dir: Path = DATASETS_DIR
) -> List[Optional[int]]:
	"""
	Grab the OBSERVED genotype vector for one individual (a single column).

	Returns a Python list aligned to sites (row order in the CSV):
	  - 0/1/2 as ints
	  - None where the observed value is missing

	If observed_df is provided, avoids re-reading the CSV (much faster when called repeatedly).
	"""
	if observed_df is None:
		observed_path = datasets_dir / f'{dataset_name}.observed_genotypes.csv'
		if not observed_path.exists():
			print('error in get_genetic_data_from_id')
			raise FileNotFoundError(f'Missing file: {observed_path}')
		observed_df = pd.read_csv(observed_path)

	col = col_name(individual_id)
	if col not in observed_df.columns:
		available = [c for c in observed_df.columns if c.startswith('i_')]
		raise KeyError(f"Column '{col}' not found. Available individual columns: {available[:5]}{'...' if len(available) > 5 else ''}")

	series = observed_df[col]

	# Convert NaN -> None, numeric -> int
	out: List[Optional[int]] = []
	for v in series.tolist():
		if pd.isna(v):
			out.append(None)
		else:
			out.append(int(v))
	return out


def get_individual_family_tree_data(dataset_name: str, individual_id: int, datasets_dir: Path = DATASETS_DIR) -> Dict[str, Any]:
	"""
	Build the family tree (connected component) around `individual_id` using pedigree.csv
	and attach observed genotype vectors for every individual in that component.

	Returns:
		{
			"dataset": str,
			"focus_id": int,
			"nodes": [
				{"id": int, "time": int|float, "observed": [0|1|2|None, ...]},
				...
			],
			"edges": [
				{"source": parent_id, "target": child_id},
				...
			]
		}
	"""
	paths = _paths_for_dataset(dataset_name, datasets_dir=datasets_dir)
	pedigree_path = paths['pedigree_csv']

	if not pedigree_path.exists():
		raise FileNotFoundError(f'Missing file: {pedigree_path}')

	ped = pd.read_csv(pedigree_path)

	required = {'individual_id', 'time', 'parent_0_id', 'parent_1_id'}
	missing = required - set(ped.columns)
	if missing:
		raise ValueError(f'Pedigree CSV missing columns: {sorted(missing)}')

	# Normalize to int where appropriate
	ped['individual_id'] = ped['individual_id'].astype(int)
	ped['parent_0_id'] = ped['parent_0_id'].astype(int)
	ped['parent_1_id'] = ped['parent_1_id'].astype(int)

	# Quick lookup maps
	parents_of: Dict[int, Tuple[int, int]] = {}
	time_of: Dict[int, float] = {}
	children_of: Dict[int, List[int]] = {}

	for _, r in ped.iterrows():
		child = int(r['individual_id'])
		p0 = int(r['parent_0_id'])
		p1 = int(r['parent_1_id'])
		t = float(r['time'])

		parents_of[child] = (p0, p1)
		time_of[child] = t

		if p0 != -1:
			children_of.setdefault(p0, []).append(child)
		if p1 != -1:
			children_of.setdefault(p1, []).append(child)

	print('finished for loop')

	if individual_id not in time_of:
		raise KeyError(f'individual_id {individual_id} not found in {pedigree_path.name}')

	# Connected component via DFS
	component: Set[int] = set()
	stack: List[int] = [individual_id]

	while stack:
		cur = stack.pop()
		if cur in component:
			continue
		component.add(cur)

		p0, p1 = parents_of.get(cur, (-1, -1))
		if p0 != -1:
			stack.append(p0)
		if p1 != -1:
			stack.append(p1)

		for ch in children_of.get(cur, []):
			stack.append(ch)

	# Build edges restricted to this component
	edges: List[Dict[str, int]] = []
	for child in component:
		p0, p1 = parents_of.get(child, (-1, -1))
		if p0 != -1 and p0 in component:
			edges.append({'source': p0, 'target': child})
		if p1 != -1 and p1 in component:
			edges.append({'source': p1, 'target': child})

	print('finished the stack for DFS')

	# Read observed genotypes ONCE
	observed_path = paths['observed_csv']
	if not observed_path.exists():
		raise FileNotFoundError(f'Missing file: {observed_path}')
	observed_df = pd.read_csv(observed_path)

	# Build nodes with cached observed_df
	nodes: List[Dict[str, Any]] = []
	for ind in sorted(component):
		observed_vec = get_genetic_data_from_id(dataset_name, ind, observed_df=observed_df)
		nodes.append({'id': ind, 'time': time_of.get(ind, None), 'observed': observed_vec})

	print('ready to return')

	return {'dataset': dataset_name, 'focus_id': individual_id, 'nodes': nodes, 'edges': edges}
