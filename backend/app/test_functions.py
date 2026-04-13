import base64
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

# Add app directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from functions import (
	DashboardFilesMissing,
	api_error,
	api_success,
	col_name,
	get_all_dataset_files,
	get_dataset_dashboard_files,
	get_dataset_names,
	get_file,
	get_genetic_data_from_id,
	get_individual_family_tree_data,
	get_model_list,
)


class TestAPIHelpers:
	"""Test API response helper functions"""

	def test_api_success(self):
		"""Test successful API response format"""
		response = api_success('Test message', {'key': 'value'}, status_code=200)
		assert response.status_code == 200
		assert response.body == b'{"status":"success","message":"Test message","data":{"key":"value"}}'

	def test_api_success_custom_status(self):
		"""Test API success with custom status code"""
		response = api_success('Created', {}, status_code=201)
		assert response.status_code == 201

	def test_api_error(self):
		"""Test error API response format"""
		response = api_error('Something went wrong', status_code=400, code='bad_request')
		assert response.status_code == 400
		assert response.body == b'{"status":"error","code":"bad_request","message":"Something went wrong"}'

	def test_api_error_default_code(self):
		"""Test error response with default error code"""
		response = api_error('Server error', status_code=500)
		assert response.status_code == 500


class TestDashboardFilesMissing:
	"""Test custom exception class"""

	def test_exception_init(self):
		"""Test DashboardFilesMissing exception"""
		exc = DashboardFilesMissing('test_dataset', ['file1.csv', 'file2.csv'])
		assert exc.dataset_name == 'test_dataset'
		assert exc.missing == ['file1.csv', 'file2.csv']
		assert 'test_dataset' in str(exc)
		assert 'file1.csv' in str(exc)


class TestColName:
	"""Test column name generation"""

	def test_col_name_zero(self):
		"""Test column name for individual 0"""
		assert col_name(0) == 'i_0000'

	def test_col_name_positive(self):
		"""Test column name for positive IDs"""
		assert col_name(1) == 'i_0001'
		assert col_name(42) == 'i_0042'
		assert col_name(9999) == 'i_9999'

	def test_col_name_large(self):
		"""Test column name for large IDs"""
		assert col_name(100000) == 'i_100000'

	def test_col_name_negative_raises(self):
		"""Test that negative IDs raise ValueError"""
		with pytest.raises(ValueError, match='individual_id must be >= 0'):
			col_name(-1)


class TestGetDatasetNames:
	"""Test dataset name retrieval"""

	def test_returns_names_in_order(self):
		"""Names are returned in file order"""
		with tempfile.TemporaryDirectory() as tmpdir:
			datasets_dir = Path(tmpdir)
			(datasets_dir / 'datasets.txt').write_text('alpha\nbeta\ngamma\n')
			with patch('functions.DATASETS_DIR', datasets_dir):
				result = get_dataset_names()
			assert result == ['alpha', 'beta', 'gamma']

	def test_deduplicates_while_preserving_order(self):
		"""Duplicate entries are removed while keeping first occurrence"""
		with tempfile.TemporaryDirectory() as tmpdir:
			datasets_dir = Path(tmpdir)
			(datasets_dir / 'datasets.txt').write_text('ds1\nds2\nds1\nds3\nds2\n')
			with patch('functions.DATASETS_DIR', datasets_dir):
				result = get_dataset_names()
			assert result == ['ds1', 'ds2', 'ds3']

	def test_skips_blank_lines_and_comments(self):
		"""Blank lines and lines starting with '#' are ignored"""
		with tempfile.TemporaryDirectory() as tmpdir:
			datasets_dir = Path(tmpdir)
			(datasets_dir / 'datasets.txt').write_text('# header\n\nds1\n\n# note\nds2\n')
			with patch('functions.DATASETS_DIR', datasets_dir):
				result = get_dataset_names()
			assert result == ['ds1', 'ds2']

	def test_strips_whitespace(self):
		"""Leading/trailing whitespace on each line is stripped"""
		with tempfile.TemporaryDirectory() as tmpdir:
			datasets_dir = Path(tmpdir)
			(datasets_dir / 'datasets.txt').write_text('  ds1  \n\tds2\t\n')
			with patch('functions.DATASETS_DIR', datasets_dir):
				result = get_dataset_names()
			assert result == ['ds1', 'ds2']

	def test_returns_empty_list_when_file_missing(self):
		"""Returns [] when datasets.txt does not exist"""
		with tempfile.TemporaryDirectory() as tmpdir:
			datasets_dir = Path(tmpdir)  # no datasets.txt created
			with patch('functions.DATASETS_DIR', datasets_dir):
				result = get_dataset_names()
			assert result == []


class TestGetModelList:
	"""Test model list retrieval"""

	def test_get_model_list(self):
		"""Test reading models from CSV"""
		with tempfile.TemporaryDirectory() as tmpdir:
			models_dir = Path(tmpdir)
			csv_path = models_dir / 'models.csv'

			csv_content = """model_name,model_type
testing.training,bayes_softmax3
testing.training,multi_log_regression
test_prep.training,dnn"""

			csv_path.write_text(csv_content)

			models = get_model_list(models_dir)

			assert len(models) == 3
			assert {'model_name': 'testing.training', 'model_type': 'bayes_softmax3'} in models
			assert {'model_name': 'test_prep.training', 'model_type': 'dnn'} in models

	def test_get_model_list_missing_file(self):
		"""Test error when models.csv is missing"""
		with tempfile.TemporaryDirectory() as tmpdir:
			models_dir = Path(tmpdir)

			with pytest.raises(FileNotFoundError, match='models.csv not found'):
				get_model_list(models_dir)


class TestGetFile:
	"""Test file reading"""

	def test_get_file_text_mode(self):
		"""Test reading file as text"""
		with tempfile.TemporaryDirectory() as tmpdir:
			file_path = Path(tmpdir) / 'test.txt'
			file_path.write_text('Hello, World!')

			result = get_file(file_path, mode='text')

			assert result['name'] == 'test.txt'
			assert result['text'] == 'Hello, World!'

	def test_get_file_base64_mode(self):
		"""Test reading file as base64"""
		with tempfile.TemporaryDirectory() as tmpdir:
			file_path = Path(tmpdir) / 'test.bin'
			content = b'Binary content'
			file_path.write_bytes(content)

			result = get_file(file_path, mode='base64')

			assert result['name'] == 'test.bin'
			assert result['base64'] == base64.b64encode(content).decode('ascii')
			assert result['byte_length'] == len(content)

	def test_get_file_missing(self):
		"""Test error when file doesn't exist"""
		nonexistent = Path('/tmp/nonexistent_file_12345.txt')

		with pytest.raises(FileNotFoundError):
			get_file(nonexistent)

	def test_get_file_encoding(self):
		"""Test reading file with specific encoding"""
		with tempfile.TemporaryDirectory() as tmpdir:
			file_path = Path(tmpdir) / 'test_utf8.txt'
			content = 'Héllo Wørld'
			file_path.write_text(content, encoding='utf-8')

			result = get_file(file_path, mode='text', encoding='utf-8')

			assert result['text'] == content


class TestGetDatasetDashboardFiles:
	"""Test dashboard file retrieval"""

	def test_get_dataset_dashboard_files(self):
		"""Test getting dashboard files for a dataset"""
		with tempfile.TemporaryDirectory() as tmpdir:
			datasets_dir = Path(tmpdir)

			truth_path = datasets_dir / 'test.truth_genotypes.csv'
			observed_path = datasets_dir / 'test.observed_genotypes.csv'

			truth_path.write_text('id,value\n1,2\n')
			observed_path.write_text('id,value\n0,1\n')

			result = get_dataset_dashboard_files('test', datasets_dir)

			assert result['dataset'] == 'test'
			assert 'observed_genotypes_csv' in result
			assert 'truth_genotypes_csv' in result
			assert '0,1' in result['observed_genotypes_csv']

	def test_get_dataset_dashboard_files_missing_truth(self):
		"""Test error when truth genotypes file is missing"""
		with tempfile.TemporaryDirectory() as tmpdir:
			datasets_dir = Path(tmpdir)
			observed_path = datasets_dir / 'test.observed_genotypes.csv'
			observed_path.write_text('data')

			with pytest.raises(DashboardFilesMissing) as exc_info:
				get_dataset_dashboard_files('test', datasets_dir)

			assert 'test' in str(exc_info.value)

	def test_get_dataset_dashboard_files_missing_observed(self):
		"""Test error when observed genotypes file is missing"""
		with tempfile.TemporaryDirectory() as tmpdir:
			datasets_dir = Path(tmpdir)
			truth_path = datasets_dir / 'test.truth_genotypes.csv'
			truth_path.write_text('data')

			with pytest.raises(DashboardFilesMissing):
				get_dataset_dashboard_files('test', datasets_dir)


class TestGetAllDatasetFiles:
	"""Test all dataset file retrieval"""

	def test_get_all_dataset_files(self):
		"""Test getting all files for a dataset"""
		with tempfile.TemporaryDirectory() as tmpdir:
			datasets_dir = Path(tmpdir)

			trees_bytes = b'tree data'
			(datasets_dir / 'test.trees').write_bytes(trees_bytes)
			(datasets_dir / 'test.truth_genotypes.csv').write_text('truth,data')
			(datasets_dir / 'test.observed_genotypes.csv').write_text('observed,data')
			(datasets_dir / 'test.pedigree.csv').write_text('pedigree,data')
			(datasets_dir / 'test.run_metadata.json').write_text('{"meta": "data"}')

			result = get_all_dataset_files('test', datasets_dir)

			assert result['dataset'] == 'test'
			assert result['trees_name'] == 'test.trees'
			assert result['trees_byte_length'] == len(trees_bytes)
			assert result['trees_base64'] == base64.b64encode(trees_bytes).decode('ascii')
			assert result['truth_genotypes_csv'] == 'truth,data'
			assert result['observed_genotypes_csv'] == 'observed,data'
			assert result['pedigree_csv'] == 'pedigree,data'
			assert result['run_metadata_json'] == '{"meta": "data"}'

	def test_get_all_dataset_files_missing(self):
		"""Test error when files are missing"""
		with tempfile.TemporaryDirectory() as tmpdir:
			datasets_dir = Path(tmpdir)
			(datasets_dir / 'test.trees').write_bytes(b'tree')

			with pytest.raises(DashboardFilesMissing):
				get_all_dataset_files('test', datasets_dir)


class TestGetGeneticDataFromId:
	"""Test genetic data retrieval"""

	def test_get_genetic_data_from_id(self):
		"""Test retrieving genetic data for an individual"""
		df = pd.DataFrame({'i_0000': [0, 1, 2, None], 'i_0001': [1, 1, 0, 2]})

		result = get_genetic_data_from_id('test', 0, observed_df=df)

		assert result == [0, 1, 2, None]

	def test_get_genetic_data_from_id_with_nans(self):
		"""Test that NaN values are converted to None"""
		df = pd.DataFrame({'i_0000': [0.0, float('nan'), 2.0]})

		result = get_genetic_data_from_id('test', 0, observed_df=df)

		assert result[0] == 0
		assert result[1] is None
		assert result[2] == 2

	def test_get_genetic_data_from_id_missing_column(self):
		"""Test error when column doesn't exist"""
		df = pd.DataFrame({'i_0001': [0, 1, 2]})

		with pytest.raises(KeyError, match='i_0000'):
			get_genetic_data_from_id('test', 0, observed_df=df)

	def test_get_genetic_data_from_id_from_file(self):
		"""Test reading from file when dataframe not provided"""
		with tempfile.TemporaryDirectory() as tmpdir:
			datasets_dir = Path(tmpdir)
			csv_path = datasets_dir / 'test.observed_genotypes.csv'

			df = pd.DataFrame({'i_0000': [0, 1, 2], 'i_0001': [1, 1, 0]})
			df.to_csv(csv_path, index=False)

			result = get_genetic_data_from_id('test', 0, datasets_dir=datasets_dir)

			assert result == [0, 1, 2]

	def test_get_genetic_data_from_id_file_not_found(self):
		"""FileNotFoundError when no df and the CSV doesn't exist"""
		with tempfile.TemporaryDirectory() as tmpdir:
			datasets_dir = Path(tmpdir)  # no CSV created
			with pytest.raises(FileNotFoundError):
				get_genetic_data_from_id('nonexistent', 0, datasets_dir=datasets_dir)


class TestGetIndividualFamilyTreeData:
	"""Test family tree data retrieval"""

	def test_get_individual_family_tree_data(self):
		"""Test building family tree for an individual"""
		with tempfile.TemporaryDirectory() as tmpdir:
			datasets_dir = Path(tmpdir)

			# Create pedigree CSV
			pedigree_path = datasets_dir / 'test.pedigree.csv'
			pedigree_df = pd.DataFrame({'individual_id': [0, 1, 2], 'time': [0, 1, 2], 'parent_0_id': [-1, 0, 0], 'parent_1_id': [-1, -1, 1]})
			pedigree_df.to_csv(pedigree_path, index=False)

			# Create observed genotypes CSV
			observed_path = datasets_dir / 'test.observed_genotypes.csv'
			observed_df = pd.DataFrame({'i_0000': [0, 1, 2], 'i_0001': [1, 1, 0], 'i_0002': [0, 0, 1]})
			observed_df.to_csv(observed_path, index=False)

			result = get_individual_family_tree_data('test', 2, datasets_dir)

			assert result['dataset'] == 'test'
			assert result['focus_id'] == 2
			assert 'nodes' in result
			assert 'edges' in result
			assert len(result['nodes']) > 0

	def test_get_individual_family_tree_data_missing_individual(self):
		"""Test error when individual not in pedigree"""
		with tempfile.TemporaryDirectory() as tmpdir:
			datasets_dir = Path(tmpdir)

			pedigree_path = datasets_dir / 'test.pedigree.csv'
			pedigree_df = pd.DataFrame({'individual_id': [0, 1], 'time': [0, 1], 'parent_0_id': [-1, 0], 'parent_1_id': [-1, -1]})
			pedigree_df.to_csv(pedigree_path, index=False)

			observed_path = datasets_dir / 'test.observed_genotypes.csv'
			observed_df = pd.DataFrame({'i_0000': [0, 1], 'i_0001': [1, 0]})
			observed_df.to_csv(observed_path, index=False)

			with pytest.raises(KeyError, match='individual_id 999 not found'):
				get_individual_family_tree_data('test', 999, datasets_dir)

	def test_get_individual_family_tree_data_missing_pedigree(self):
		"""Test error when pedigree file is missing"""
		with tempfile.TemporaryDirectory() as tmpdir:
			datasets_dir = Path(tmpdir)

			with pytest.raises(FileNotFoundError):
				get_individual_family_tree_data('test', 0, datasets_dir)

	def test_get_individual_family_tree_data_missing_observed_file(self):
		"""FileNotFoundError when the observed genotypes CSV is absent"""
		with tempfile.TemporaryDirectory() as tmpdir:
			datasets_dir = Path(tmpdir)

			ped_df = pd.DataFrame({'individual_id': [0], 'time': [0.0], 'parent_0_id': [-1], 'parent_1_id': [-1]})
			ped_df.to_csv(datasets_dir / 'test.pedigree.csv', index=False)
			# observed_genotypes.csv intentionally omitted

			with pytest.raises(FileNotFoundError):
				get_individual_family_tree_data('test', 0, datasets_dir)

	def test_get_individual_family_tree_data_missing_pedigree_columns(self):
		"""ValueError when required columns are absent from the pedigree CSV"""
		with tempfile.TemporaryDirectory() as tmpdir:
			datasets_dir = Path(tmpdir)

			# Pedigree missing parent columns
			ped_df = pd.DataFrame({'individual_id': [0, 1], 'time': [1.0, 0.0]})
			ped_df.to_csv(datasets_dir / 'test.pedigree.csv', index=False)

			with pytest.raises(ValueError, match='missing columns'):
				get_individual_family_tree_data('test', 0, datasets_dir)

	def test_get_individual_family_tree_data_node_and_edge_structure(self):
		"""Returned nodes and edges have the expected keys"""
		with tempfile.TemporaryDirectory() as tmpdir:
			datasets_dir = Path(tmpdir)

			ped_df = pd.DataFrame({'individual_id': [0, 1, 2], 'time': [2.0, 1.0, 0.0], 'parent_0_id': [-1, 0, 0], 'parent_1_id': [-1, -1, 1]})
			ped_df.to_csv(datasets_dir / 'test.pedigree.csv', index=False)

			obs_df = pd.DataFrame({'i_0000': [0, 1], 'i_0001': [1, 0], 'i_0002': [0, 1]})
			obs_df.to_csv(datasets_dir / 'test.observed_genotypes.csv', index=False)

			result = get_individual_family_tree_data('test', 2, datasets_dir)

			# Every node must have id, time, and observed keys
			for node in result['nodes']:
				assert 'id' in node, f'Node missing id: {node}'
				assert 'time' in node, f'Node missing time: {node}'
				assert 'observed' in node, f'Node missing observed: {node}'
				assert isinstance(node['observed'], list)

			# Every edge must have source and target keys
			for edge in result['edges']:
				assert 'source' in edge, f'Edge missing source: {edge}'
				assert 'target' in edge, f'Edge missing target: {edge}'
