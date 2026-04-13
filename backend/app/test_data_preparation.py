"""
Test suite for data_preparation.py module.
Tests pedigree graph construction, feature engineering, and data preparation pipelines.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from dotenv import load_dotenv

# Handle relative imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.data_generation import SimConfig, run_generation
from app.data_preparation import (
	PrepConfig,
	build_adjacency_from_pedigree,
	build_split_examples,
	connected_components,
	genotype_columns,
	is_fully_missing_individual,
	k_hop_neighborhood,
	make_example_for_target,
	prepare_data,
	resample_training_data,
)

load_dotenv()

DATASETS_DIR = os.getenv('DATASETS_DIR')


class TestDataPreparation:
	"""Test suite for data_preparation.py module"""

	@staticmethod
	def setup_test_dataset(tmpdir: str = None):
		"""
		Generate a small test dataset for use in all tests.
		Returns (truth_df, obs_df, ped_df, outputs_dict, tmpdir)
		"""
		if tmpdir is None:
			tmpdir = DATASETS_DIR
		cfg = SimConfig(
			n_diploid_samples=50,
			sequence_length=200,
			n_generations=4,
			samples_per_generation=15,
			seed=9999,
			masking_rate=0.25,
			datasets_dir=tmpdir,
			full_data=False,
			name='test_prep',
		)

		outputs = run_generation(cfg)

		truth = pd.read_csv(outputs['truth_csv'])
		obs = pd.read_csv(outputs['observed_csv'])
		ped = pd.read_csv(outputs['pedigree_csv'])

		return truth, obs, ped, outputs, tmpdir

	def test_genotype_columns(self):
		"""Test genotype_columns() helper"""
		print('=' * 60)
		print('Test: genotype_columns()')
		print('=' * 60)

		df = pd.DataFrame({'index': [0, 1, 2], 'i_0000': [0, 1, 2], 'i_0001': [1, 1, 0], 'other_col': [5, 6, 7], 'i_0002': [0, 0, 1]})

		cols = genotype_columns(df)
		assert set(cols) == {'i_0000', 'i_0001', 'i_0002'}, f'Expected i_* columns, got {cols}'
		print(f'✓ Found columns: {cols}')
		print()

	def test_genotype_columns_empty(self):
		"""genotype_columns on a DataFrame with no i_* columns returns empty list"""
		df = pd.DataFrame({'index': [0], 'other': [1]})
		assert genotype_columns(df) == []

	def test_pedigree_adjacency(self):
		"""Test building adjacency list from pedigree"""
		print('=' * 60)
		print('Test: build_adjacency_from_pedigree()')
		print('=' * 60)

		ped = pd.DataFrame(
			{
				'individual_id': [0, 1, 2, 3, 4],
				'parent_0_id': [-1, -1, 0, 0, 1],
				'parent_1_id': [-1, -1, 1, 1, 2],
				'time': [3.0, 3.0, 2.0, 2.0, 1.0],
				'num_nodes': [2, 2, 2, 2, 2],
			}
		)

		adj = build_adjacency_from_pedigree(ped)

		# 0 and 1 are founders
		# 2, 3 have parents 0, 1
		# 4 has parents 1, 2
		assert 2 in adj[0], 'Edge 0-2 missing'
		assert 3 in adj[0], 'Edge 0-3 missing'
		assert 2 in adj[1], 'Edge 1-2 missing'
		assert 3 in adj[1], 'Edge 1-3 missing'
		assert 4 in adj[1], 'Edge 1-4 missing'
		assert 4 in adj[2], 'Edge 2-4 missing'

		print(f'✓ Adjacency list built correctly for {len(adj)} individuals')
		print(f'  Sample node 0 neighbors: {adj[0]}')
		print()

	def test_pedigree_adjacency_missing_columns(self):
		"""build_adjacency_from_pedigree should raise ValueError when required columns are absent"""
		ped = pd.DataFrame({'individual_id': [0, 1], 'time': [1.0, 0.0]})
		with pytest.raises(ValueError, match='missing columns'):
			build_adjacency_from_pedigree(ped)

	def test_connected_components(self):
		"""Test finding connected components in pedigree graph"""
		print('=' * 60)
		print('Test: connected_components()')
		print('=' * 60)

		# Two separate families
		ped = pd.DataFrame(
			{
				'individual_id': [0, 1, 2, 3, 4, 5],
				'parent_0_id': [-1, -1, 0, 0, -1, 4],
				'parent_1_id': [-1, -1, 1, 1, -1, -1],
				'time': [2.0, 2.0, 1.0, 1.0, 2.0, 1.0],
				'num_nodes': [2, 2, 2, 2, 2, 2],
			}
		)

		adj = build_adjacency_from_pedigree(ped)
		comps = connected_components(adj)

		assert len(comps) == 2, f'Expected 2 components, got {len(comps)}'
		comp_sizes = sorted([len(c) for c in comps])
		assert comp_sizes == [2, 4], f'Expected sizes [2, 4], got {comp_sizes}'

		print(f'✓ Found {len(comps)} connected components')
		for i, comp in enumerate(comps):
			print(f'  Component {i + 1}: {len(comp)} individuals: {sorted(comp)}')
		print()

	def test_connected_components_single_isolated_node(self):
		"""A pedigree with one individual and no parents produces one component of size 1"""
		ped = pd.DataFrame({'individual_id': [0], 'parent_0_id': [-1], 'parent_1_id': [-1], 'time': [0.0], 'num_nodes': [2]})
		adj = build_adjacency_from_pedigree(ped)
		comps = connected_components(adj)
		assert len(comps) == 1
		assert 0 in comps[0]

	def test_k_hop_neighborhood(self):
		"""Test k-hop neighborhood computation"""
		print('=' * 60)
		print('Test: k_hop_neighborhood()')
		print('=' * 60)

		ped = pd.DataFrame(
			{
				'individual_id': [0, 1, 2, 3, 4, 5],
				'parent_0_id': [-1, -1, 0, 0, 1, 2],
				'parent_1_id': [-1, -1, 1, 1, 2, 3],
				'time': [3.0, 3.0, 2.0, 2.0, 1.0, 0.0],
				'num_nodes': [2, 2, 2, 2, 2, 2],
			}
		)

		adj = build_adjacency_from_pedigree(ped)

		# From node 2:
		# Direct connections (parent-child edges): {0, 1, 4, 5}
		# 1-hop from 2: {0, 1, 4, 5}
		# 2-hop from 2: {0, 1, 3, 4, 5} (adds 3 via 0 or 1)
		neighbors_1 = k_hop_neighborhood(adj, 2, 1)
		neighbors_2 = k_hop_neighborhood(adj, 2, 2)

		assert 0 in neighbors_1 and 1 in neighbors_1, '1-hop should include parents'
		assert 4 in neighbors_1 and 5 in neighbors_1, 'Children should be in 1-hop from 2'
		assert 3 not in neighbors_1, '3 should NOT be in 1-hop (not directly connected)'
		assert 3 in neighbors_2, '3 should be in 2-hop (via parent 0 or 1)'

		print(f'✓ 1-hop neighbors of 2: {sorted(neighbors_1)}')
		print(f'✓ 2-hop neighbors of 2: {sorted(neighbors_2)}')
		print()

	def test_k_hop_neighborhood_k_zero(self):
		"""k=0 should always return the empty set"""
		adj = {0: {1, 2}, 1: {0}, 2: {0}}
		assert k_hop_neighborhood(adj, 0, 0) == set()

	def test_k_hop_neighborhood_isolated_node(self):
		"""A node with no neighbours returns an empty set regardless of k"""
		adj = {0: set()}
		assert k_hop_neighborhood(adj, 0, 3) == set()

	def test_is_fully_missing_individual(self):
		"""Test detection of fully masked individuals"""
		print('=' * 60)
		print('Test: is_fully_missing_individual()')
		print('=' * 60)

		obs = pd.DataFrame(
			{
				'index': [0, 1, 2],
				'i_0000': [0.0, 1.0, 2.0],
				'i_0001': [np.nan, np.nan, np.nan],  # fully masked
				'i_0002': [1.0, np.nan, 0.0],  # partially masked
			}
		)

		assert not is_fully_missing_individual(obs, 0), 'i_0000 should not be fully masked'
		assert is_fully_missing_individual(obs, 1), 'i_0001 should be fully masked'
		assert not is_fully_missing_individual(obs, 2), 'i_0002 should not be fully masked'

		# Column not present in df should return False (not a masked target)
		assert not is_fully_missing_individual(obs, 99), 'Missing column should return False'

		print('✓ Individual 0 (i_0000): observed')
		print('✓ Individual 1 (i_0001): fully masked')
		print('✓ Individual 2 (i_0002): partially masked')
		print()

	def test_make_example_for_target(self):
		"""Test feature generation for a single target individual"""
		print('=' * 60)
		print('Test: make_example_for_target()')
		print('=' * 60)

		truth, obs, ped, _, _ = self.setup_test_dataset(DATASETS_DIR)

		adj = build_adjacency_from_pedigree(ped)

		# Find a target with relatives
		for target_id in ped['individual_id'].values:
			ex = make_example_for_target(truth, obs, adj, target_id, max_hops=2)
			if ex is not None:
				X, y = ex
				assert X.shape[1] == 3, f'X should have 3 features, got {X.shape[1]}'
				assert X.shape[0] == truth.shape[0], 'X rows should match sites'
				assert y.shape[0] == truth.shape[0], 'y rows should match sites'
				print(f'✓ Generated features for individual {target_id}')
				print(f'  X shape: {X.shape}, y shape: {y.shape}')
				print(
					f'  Feature ranges: mean_dosage [{X[:, 0].min():.2f}, {X[:, 0].max():.2f}], '
					f'frac_observed [{X[:, 1].min():.2f}, {X[:, 1].max():.2f}], '
					f'count_relatives [{int(X[0, 2])}]'
				)
				break

		print()

	def test_build_split_examples(self):
		"""Test building training examples from families"""
		print('=' * 60)
		print('Test: build_split_examples()')
		print('=' * 60)

		truth, obs, ped, _, _ = self.setup_test_dataset(DATASETS_DIR)

		adj = build_adjacency_from_pedigree(ped)
		comps = connected_components(adj)

		cfg = PrepConfig(
			dataset_name='test',
			max_hops=2,
			only_predict_masked=False,  # Include all individuals
		)

		X, y, g, target_ids = build_split_examples(truth, obs, adj, comps, cfg, ped)

		assert X.ndim == 3, f'X should be 3D, got shape {X.shape}'
		assert X.shape[1] == truth.shape[0], f'X should have {truth.shape[0]} sites'
		assert X.shape[2] == 3, 'X should have 3 features'
		assert y.shape[0] == X.shape[0], 'y should have same number of examples as X'
		assert y.shape[1] == truth.shape[0], 'y should have same number of sites as X'
		assert g.shape[0] == X.shape[0], 'groups should have same length as examples'
		assert len(target_ids) == X.shape[0], 'target_ids length should match example count'
		assert all(isinstance(tid, (int, np.integer)) for tid in target_ids), 'target_ids should be integers'

		print(f'✓ Built examples from {len(comps)} families')
		print(f'  X shape: {X.shape}')
		print(f'  y shape: {y.shape}')
		print(f'  groups shape: {g.shape}')
		print(f'  target_ids (first 5): {target_ids[:5]}')
		print(f'  Generation times: min={g.min()}, max={g.max()}')
		print()

	def test_build_split_examples_empty_when_no_masked(self):
		"""Returns empty arrays with correct ranks when only_predict_masked=True and nobody is masked"""
		truth = pd.DataFrame({'index': [0, 1], 'i_0000': [0, 1], 'i_0001': [1, 0]})
		obs = pd.DataFrame({'index': [0, 1], 'i_0000': [0.0, 1.0], 'i_0001': [1.0, 0.0]})  # fully observed
		ped = pd.DataFrame({'individual_id': [0, 1], 'parent_0_id': [-1, 0], 'parent_1_id': [-1, -1], 'time': [1.0, 0.0], 'num_nodes': [2, 2]})
		adj = build_adjacency_from_pedigree(ped)
		comps = connected_components(adj)
		cfg = PrepConfig(dataset_name='test', only_predict_masked=True)
		X, y, g, ids = build_split_examples(truth, obs, adj, comps, cfg, ped)
		assert X.ndim == 3
		assert X.shape[0] == 0, 'Should produce no examples when none are masked'

	def test_resample_training_data(self):
		"""Test random oversampling for class balance"""
		print('=' * 60)
		print('Test: resample_training_data()')
		print('=' * 60)

		# Create imbalanced data: class 0 has few samples
		X = np.random.randn(100, 50, 3)
		y = np.array([0] * 10 + [1] * 40 + [2] * 50)
		groups = np.array([0] * 100)

		np.random.seed(42)
		X_resampled, y_resampled, g_resampled = resample_training_data(X, y, groups)

		unique, counts = np.unique(y_resampled, return_counts=True)
		print(f'✓ Original class distribution: {dict(zip(*np.unique(y, return_counts=True)))}')
		print(f'✓ Resampled class distribution: {dict(zip(unique, counts))}')
		print(f'  All classes now have {counts[0]} samples')
		assert all(c == counts[0] for c in counts), 'All classes should have same count after resampling'
		print()

	def test_prepare_data_returns_expected_keys(self):
		"""prepare_data() returns a dict with X, y, and groups arrays."""
		print('=' * 60)
		print('Test: prepare_data()')
		print('=' * 60)

		truth, obs, ped, _, tmpdir = self.setup_test_dataset(DATASETS_DIR)

		cfg = PrepConfig(dataset_name='test_prep', max_hops=2, only_predict_masked=True, datasets_dir=tmpdir)

		result = prepare_data(cfg)

		assert set(result.keys()) >= {'X', 'y', 'groups'}, f'Missing keys in result: {result.keys()}'
		assert isinstance(result['X'], np.ndarray), 'X should be ndarray'
		assert isinstance(result['y'], np.ndarray), 'y should be ndarray'
		assert isinstance(result['groups'], np.ndarray), 'groups should be ndarray'
		assert result['X'].ndim == 3, 'X should be 3D (examples x sites x features)'
		assert result['y'].ndim == 2, 'y should be 2D (examples x sites)'
		assert result['X'].shape[0] == result['y'].shape[0], 'X and y example counts must match'
		print(f'✓ prepare_data() returned X={result["X"].shape}, y={result["y"].shape}')
		print()

	def test_prep_config_defaults(self):
		"""PrepConfig has expected defaults"""
		cfg = PrepConfig(dataset_name='dummy')
		assert cfg.train_frac == 0.70
		assert cfg.val_frac == 0.15
		assert cfg.test_frac == 0.15
		assert cfg.max_hops == 2
		assert cfg.only_predict_masked is True
		assert cfg.datasets_dir is None

	def test_prep_config_frozen(self):
		"""PrepConfig is frozen — mutation should raise"""
		cfg = PrepConfig(dataset_name='dummy')
		with pytest.raises((AttributeError, TypeError)):
			cfg.seed = 0  # type: ignore[misc]
