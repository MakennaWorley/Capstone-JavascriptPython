import os

import numpy as np
import pandas as pd
import pytest
import tskit
from dotenv import load_dotenv

from .data_generation import (
	MAX_USER_GENERATIONS,
	MAX_USER_LENGTH,
	MAX_USER_SAMPLES,
	SimConfig,
	build_config_from_params,
	config_to_dict,
	dataset_exists,
	dict_to_config,
	get_preset_config,
	mask,
	run_generation,
	validate_api_request,
)

load_dotenv()

DATASETS_DIR = os.getenv('DATASETS_DIR')


class TestSimConfig:
	"""Tests for SimConfig defaults and construction"""

	def test_default_values(self):
		"""Test that SimConfig has the expected default values"""
		cfg = SimConfig()
		assert cfg.n_diploid_samples == 250
		assert cfg.sequence_length == 100
		assert cfg.n_generations == 5
		assert cfg.masking_rate == 0.20
		assert cfg.seed == 42
		assert cfg.full_data is False

	def test_custom_values(self):
		"""Test that custom values are stored correctly"""
		cfg = SimConfig(seed=999, sequence_length=200, name='custom')
		assert cfg.seed == 999
		assert cfg.sequence_length == 200
		assert cfg.name == 'custom'

	def test_frozen(self):
		"""SimConfig is frozen — mutation should raise"""
		cfg = SimConfig()
		with pytest.raises((AttributeError, TypeError)):
			cfg.seed = 99  # type: ignore[misc]


class TestGetPresetConfig:
	"""Tests for get_preset_config() preset factory"""

	def test_known_tiers_return_required_keys(self):
		for tier in ('tiny', 'small', 'medium', 'large'):
			params = get_preset_config(tier, 'base')
			assert 'n_diploid_samples' in params
			assert 'sequence_length' in params
			assert 'n_generations' in params
			assert params['name'] == f'base_{tier}'

	def test_tiers_increase_in_scale(self):
		"""Each larger tier should have bigger sample/length values"""
		tiny = get_preset_config('tiny', 'x')
		small = get_preset_config('small', 'x')
		assert small['n_diploid_samples'] >= tiny['n_diploid_samples']
		assert small['sequence_length'] >= tiny['sequence_length']

	def test_unknown_tier_falls_back_to_tiny(self):
		params = get_preset_config('nonexistent_tier', 'test')
		tiny = get_preset_config('tiny', 'test')
		assert params['n_diploid_samples'] == tiny['n_diploid_samples']

	def test_case_insensitive(self):
		lower = get_preset_config('tiny', 'x')
		upper = get_preset_config('TINY', 'x')
		assert lower['n_diploid_samples'] == upper['n_diploid_samples']


class TestMask:
	"""Unit tests for the mask() function"""

	def test_zero_masking_rate(self):
		"""At rate 0, no values should be masked"""
		rng = np.random.default_rng(42)
		X = np.ones((10, 20), dtype=np.int8)
		masked, _ = mask(X, 0.0, rng)
		assert not np.any(np.isnan(masked)), 'No values should be masked at rate 0'

	def test_full_masking_rate(self):
		"""At rate 1, all values should be masked"""
		rng = np.random.default_rng(42)
		X = np.ones((10, 20), dtype=np.int8)
		masked, _ = mask(X, 1.0, rng)
		assert np.all(np.isnan(masked)), 'All values should be masked at rate 1'

	def test_masking_is_column_wise(self):
		"""Each column must be either fully observed or fully masked — never partial"""
		rng = np.random.default_rng(42)
		X = np.ones((50, 100), dtype=np.int8)
		masked, _ = mask(X, 0.3, rng)
		for col_idx in range(masked.shape[1]):
			col_nans = np.isnan(masked[:, col_idx])
			assert col_nans.all() or (~col_nans).all(), f'Column {col_idx} has partial masking'

	def test_approximate_masking_rate(self):
		"""Masked column fraction should be close to the requested rate"""
		rng = np.random.default_rng(0)
		X = np.zeros((10, 1000), dtype=np.int8)
		masked, _ = mask(X, 0.3, rng)
		fully_masked = sum(1 for j in range(1000) if np.all(np.isnan(masked[:, j])))
		rate = fully_masked / 1000
		assert abs(rate - 0.3) < 0.05, f'Expected ~0.30 masking rate, got {rate:.3f}'

	def test_output_shapes_match_input(self):
		rng = np.random.default_rng(1)
		X = np.zeros((5, 8), dtype=np.int8)
		masked, full_mask = mask(X, 0.5, rng)
		assert masked.shape == X.shape
		assert full_mask.shape == X.shape

	def test_unmasked_values_unchanged(self):
		"""Values in observed columns must not be altered"""
		rng = np.random.default_rng(7)
		X = np.arange(50, dtype=np.float64).reshape(5, 10)
		masked, full_mask = mask(X, 0.0, rng)
		np.testing.assert_array_equal(masked, X)


class TestConfigRoundTrip:
	"""Tests for config_to_dict() / dict_to_config() serialization"""

	def test_round_trip(self):
		cfg = SimConfig(seed=777, sequence_length=200, name='rt_test')
		cfg2 = dict_to_config(config_to_dict(cfg))
		assert cfg2.seed == 777
		assert cfg2.sequence_length == 200
		assert cfg2.name == 'rt_test'

	def test_dict_to_config_ignores_unknown_keys(self):
		d = {'seed': 1, 'unknown_field': 'hello', 'name': 'test', 'sequence_length': 100}
		cfg = dict_to_config(d)
		assert cfg.seed == 1

	def test_dict_to_config_prefix_alias(self):
		"""Old-style 'prefix' key should be treated as 'name'"""
		d = {'prefix': 'my_prefix', 'seed': 1}
		cfg = dict_to_config(d)
		assert cfg.name == 'my_prefix'


class TestValidateApiRequest:
	"""Tests for validate_api_request() limit enforcement"""

	def test_within_limits(self):
		cfg = SimConfig(n_diploid_samples=100, sequence_length=500, n_generations=5)
		validate_api_request(cfg)  # Should not raise

	def test_at_exact_limits(self):
		cfg = SimConfig(n_diploid_samples=MAX_USER_SAMPLES, sequence_length=MAX_USER_LENGTH, n_generations=MAX_USER_GENERATIONS)
		validate_api_request(cfg)  # Boundary values should be allowed

	def test_exceeds_samples(self):
		cfg = SimConfig(n_diploid_samples=MAX_USER_SAMPLES + 1)
		with pytest.raises(ValueError, match='Sample size'):
			validate_api_request(cfg)

	def test_exceeds_length(self):
		cfg = SimConfig(sequence_length=MAX_USER_LENGTH + 1)
		with pytest.raises(ValueError, match='Sequence length'):
			validate_api_request(cfg)

	def test_exceeds_generations(self):
		cfg = SimConfig(n_generations=MAX_USER_GENERATIONS + 1)
		with pytest.raises(ValueError, match='Generations'):
			validate_api_request(cfg)


class TestBuildConfigFromParams:
	"""Tests for build_config_from_params() API helper"""

	def test_basic_params(self):
		params = {'sequence_length': 50, 'n_generations': 3, 'seed': 77, 'name': 'test_api', 'n_diploid_samples': 10}
		cfg = build_config_from_params(params)
		assert cfg.sequence_length == 50
		assert cfg.n_generations == 3
		assert cfg.seed == 77

	def test_auto_seed_when_missing(self):
		params = {'sequence_length': 50, 'name': 'no_seed'}
		cfg = build_config_from_params(params)
		assert cfg.seed is not None
		assert isinstance(cfg.seed, int)

	def test_auto_seed_when_none(self):
		params = {'seed': None, 'name': 'none_seed'}
		cfg = build_config_from_params(params)
		assert cfg.seed is not None

	def test_ignores_none_values(self):
		"""None values should not override SimConfig defaults"""
		params = {'sequence_length': 100, 'mutation_rate': None, 'name': 'partial', 'seed': 1}
		cfg = build_config_from_params(params)
		assert cfg.mutation_rate == SimConfig.mutation_rate


class TestDatasetExists:
	"""Tests for dataset_exists() filesystem check"""

	def test_nonexistent_directory(self, tmp_path):
		result = dataset_exists('anything', str(tmp_path))
		assert result is False

	def test_found_in_file(self, tmp_path):
		(tmp_path / 'datasets.txt').write_text('dataset_one\ndataset_two\n')
		assert dataset_exists('dataset_one', str(tmp_path)) is True
		assert dataset_exists('dataset_two', str(tmp_path)) is True

	def test_not_found_in_file(self, tmp_path):
		(tmp_path / 'datasets.txt').write_text('dataset_one\n')
		assert dataset_exists('dataset_three', str(tmp_path)) is False

	def test_empty_file(self, tmp_path):
		(tmp_path / 'datasets.txt').write_text('')
		assert dataset_exists('anything', str(tmp_path)) is False


class TestDataGeneration:
	"""Integration test suite for data generation and simulation"""

	def test_small_simulation(self):
		"""Test a small dataset generation"""
		print('=' * 60)
		print('Testing Small Simulation')
		print('=' * 60)

		cfg = SimConfig(
			n_diploid_samples=100,
			sequence_length=500,
			n_generations=5,
			samples_per_generation=20,
			seed=12345,
			masking_rate=0.15,
			datasets_dir=DATASETS_DIR,
			full_data=False,
			name='test_small',
		)

		outputs = run_generation(cfg)
		print(f'Generated outputs: {outputs}\n')

		# Load the tree sequence
		ts = tskit.load(outputs['trees'])

		print(f'Total individuals: {ts.num_individuals}')
		print(f'Total sites: {ts.num_sites}')
		print(f'Total mutations: {ts.num_mutations}')
		print(f'Total samples: {ts.num_samples}')
		print()

		# Check founder individuals (max time)
		max_time = max(ind.time for ind in ts.individuals())
		founders = [ind for ind in ts.individuals() if ind.time == max_time]
		print(f'Max generation time: {max_time}')
		print(f'Founders (n={len(founders)}): {[ind.id for ind in founders]}')
		print(f'Founder nodes: {[list(ind.nodes) for ind in founders]}')
		print()

		# Get genotype matrix
		all_nodes = []
		for ind in ts.individuals():
			if len(ind.nodes) == 2:
				all_nodes.extend([int(ind.nodes[0]), int(ind.nodes[1])])

		G = ts.genotype_matrix(samples=all_nodes, isolated_as_missing=False)
		print(f'Genotype matrix shape: {G.shape}')
		print()

		# Load and verify CSV outputs
		truth_df = pd.read_csv(outputs['truth_csv'])
		obs_df = pd.read_csv(outputs['observed_csv'])
		ped_df = pd.read_csv(outputs['pedigree_csv'])

		print(f'Truth genotypes shape: {truth_df.shape}')
		print(f'Observed genotypes shape: {obs_df.shape}')
		print(f'Pedigree table shape: {ped_df.shape}')
		print()

		# Check masking in observed
		genotype_cols = [c for c in obs_df.columns if c.startswith('i_')]
		missing_count = obs_df[genotype_cols].isna().sum().sum()
		total_genotypes = obs_df[genotype_cols].size
		masking_fraction = missing_count / total_genotypes if total_genotypes > 0 else 0
		print(f'Observed masking fraction: {masking_fraction:.3f}')
		print()

		# Check first few sites for founders
		print('Checking genotypes at first 5 sites for first 3 founders:')
		node_to_col = {node_id: col for col, node_id in enumerate(all_nodes)}

		for site_idx in range(min(5, ts.num_sites)):
			site = ts.sites()[site_idx]
			print(f'\nSite {site_idx} (position {int(site.position)}):')
			print(f'  Mutations: {[(m.node, m.derived_state) for m in site.mutations]}')

			for ind in founders[:3]:
				try:
					cols = [node_to_col[n] for n in ind.nodes]
					alleles = [G[site_idx, cols[0]], G[site_idx, cols[1]]]
					dosage = sum(alleles)
					print(f'  Ind {ind.id}: nodes {list(ind.nodes)}, alleles {alleles}, dosage {dosage}')
				except KeyError:
					print(f'  Ind {ind.id}: nodes not in sample matrix')

		# Tree sequence assertions
		assert ts.num_individuals > 0, 'No individuals generated'
		assert ts.num_sites == cfg.sequence_length, 'Tree sequence site count does not match sequence_length'
		assert G.shape[0] == cfg.sequence_length, 'Genotype matrix site count mismatch'

		# CSV shape assertions
		assert truth_df.shape[0] == cfg.sequence_length, 'Truth CSV row count should equal sequence_length'
		assert truth_df.shape == obs_df.shape, 'Truth and observed CSV shapes must match'
		assert set(truth_df.columns) == set(obs_df.columns), 'Truth and observed CSV must have identical columns'
		assert truth_df.shape[0] > 0, 'No genotypes in truth CSV'
		assert obs_df.shape[0] > 0, 'No genotypes in observed CSV'
		assert ped_df.shape[0] > 0, 'No individuals in pedigree CSV'

		# Column naming convention
		assert len(genotype_cols) > 0, 'No i_* genotype columns found in observed CSV'

		# Pedigree structure
		assert 'individual_id' in ped_df.columns, 'Pedigree missing individual_id column'
		assert 'parent_0_id' in ped_df.columns, 'Pedigree missing parent_0_id column'
		assert 'parent_1_id' in ped_df.columns, 'Pedigree missing parent_1_id column'
		assert 'time' in ped_df.columns, 'Pedigree missing time column'

		# Masking sanity: fraction of fully-masked columns should be near cfg.masking_rate
		col_fully_masked = obs_df[genotype_cols].isna().all()
		masked_col_rate = col_fully_masked.mean()
		assert masked_col_rate < cfg.masking_rate + 0.15, f'Masked column rate {masked_col_rate:.3f} is unexpectedly high'

		# Truth dosages should only contain {0, 1, 2}
		truth_genotype_vals = truth_df[genotype_cols].to_numpy().flatten()
		unique_vals = set(np.unique(truth_genotype_vals))
		assert unique_vals.issubset({0, 1, 2}), f'Unexpected dosage values in truth CSV: {unique_vals}'
