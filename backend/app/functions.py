from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

import pandas as pd
from fastapi.responses import JSONResponse

DATASETS_DIR = Path('./datasets')

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


def _col_name(individual_id: int) -> str:
	if individual_id < 0:
		raise ValueError('individual_id must be >= 0')
	return f'i_{individual_id:04d}'


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


def get_dataset_names() -> List[str]:
	"""
	Reads dataset names from a file (one per line).
	- Strips whitespace
	- Skips blank lines
	- Skips comment lines starting with '#'
	- De-duplicates while preserving order
	"""
	seen = set()
	names: List[str] = []

	base_dir = Path(__file__).resolve().parent
	file_path = base_dir / 'datasets' / 'datasets.txt'

	if not file_path.exists():
		return []

	for raw_line in file_path.read_text(encoding='utf-8').splitlines():
		line = raw_line.strip()
		if not line or line.startswith('#'):
			continue
		if line not in seen:
			seen.add(line)
			names.append(line)

	return names


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

	col = _col_name(individual_id)
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

	print(f'DEBUG: Dashboard says pedigree is at: {pedigree_path}')
	if not pedigree_path.exists():
		print('file is missing')
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
