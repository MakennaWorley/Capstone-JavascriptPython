import os

import pandas as pd
import tskit
from dotenv import load_dotenv

from .data_generation import SimConfig, run_generation

load_dotenv()

DATASETS_DIR = os.getenv('DATASETS_DIR')


class TestDataGeneration:
	"""Test suite for data generation and simulation"""

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

		# Assertions
		assert ts.num_individuals > 0, 'No individuals generated'
		assert truth_df.shape[0] > 0, 'No genotypes in truth CSV'
		assert obs_df.shape[0] > 0, 'No genotypes in observed CSV'
		assert ped_df.shape[0] > 0, 'No individuals in pedigree CSV'
		assert G.shape[0] == cfg.sequence_length, 'Genotype matrix sites mismatch'
