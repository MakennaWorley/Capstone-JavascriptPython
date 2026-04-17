"""
Test suite for model_main.py module.
Tests model training, evaluation, and utility functions with mocked dependencies.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Inject path FIRST so imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock all dependencies BEFORE importing model_main to avoid relative import errors
sys.modules.update(
	{
		'app.data_preparation': MagicMock(),
		'app.model_bayesian': MagicMock(),
		'app.model_dnn': MagicMock(),
		'app.model_functions': MagicMock(),
		'app.model_gnn': MagicMock(),
		'app.model_graph_functions': MagicMock(),
		'app.model_hmm': MagicMock(),
		'app.model_multi_log_regression': MagicMock(),
		'app.optimize_system': MagicMock(),
	}
)

# Now safely import model_main and its functions
from app.model_main import (
	OutputLogger,
	_select_model,
	check_gpu_status,
	check_model_already_applied,
	evaluate_with_cross_val,
	get_applied_models_csv_path,
	get_optimal_training_config,
	get_or_create_logs_dir,
	load_whole_dataset,
	register_model_application,
	train_eval,
	train_eval_all,
	update_models_csv,
)
from app.model_main import test_on_new_data as apply_to_new_data


class TestCheckGpuStatus:
	"""Tests for check_gpu_status function"""

	def test_check_gpu_status_runs_without_error(self, capsys):
		"""Test GPU status check completes without error"""
		# This function handles its own imports gracefully
		check_gpu_status()
		captured = capsys.readouterr()
		# Should output something about GPU or CPU
		assert len(captured.out) > 0


class TestOptimalConfig:
	"""Tests for get_optimal_training_config function"""

	def test_get_optimal_training_config_returns_dict(self, capsys):
		"""Test that config has all required keys"""
		config = get_optimal_training_config()

		assert isinstance(config, dict)
		assert 'chains' in config
		assert 'cores' in config
		assert 'draws' in config
		assert 'tune' in config
		assert 'gpu_strategy' in config
		assert config['chains'] > 0
		assert config['cores'] > 0
		assert config['draws'] > 0
		assert config['tune'] > 0


class TestSelectModel:
	"""Tests for _select_model function"""

	def test_select_model_bayes(self):
		"""Test model selection - Bayes"""
		model_cls, tag = _select_model('bayes_softmax3')
		assert tag == 'bayes_softmax3'
		assert model_cls is not None

	def test_select_model_sklearn(self):
		"""Test model selection - Sklearn"""
		model_cls, tag = _select_model('multi_log_regression')
		assert tag == 'multi_log_regression'
		assert model_cls is not None

	def test_select_model_hmm(self):
		"""Test model selection - HMM"""
		model_cls, tag = _select_model('hmm_dosage')
		assert tag == 'hmm_dosage'
		assert model_cls is not None

	def test_select_model_dnn(self):
		"""Test model selection - DNN"""
		model_cls, tag = _select_model('dnn_dosage')
		assert tag == 'dnn_dosage'
		assert model_cls is not None

	def test_select_model_gnn(self):
		"""Test model selection - GNN"""
		model_cls, tag = _select_model('gnn_dosage')
		assert tag == 'gnn_dosage'
		assert model_cls is not None

	def test_select_model_invalid(self):
		"""Test model selection with invalid model"""
		with pytest.raises(ValueError, match='Unknown model label'):
			_select_model('invalid_model')


class TestUpdateModelsCsv:
	"""Tests for update_models_csv function"""

	def test_update_models_csv_new_file(self, tmp_path):
		"""Test updating models.csv with new file"""
		csv_path = tmp_path / 'models.csv'

		update_models_csv('test.training', 'bayes_softmax3', csv_path=csv_path)

		assert csv_path.exists()
		content = csv_path.read_text()
		assert 'model_name' in content
		assert 'test.training' in content
		assert 'bayes_softmax3' in content

	def test_update_models_csv_append(self, tmp_path):
		"""Test appending to existing models.csv"""
		csv_path = tmp_path / 'models.csv'

		# Add first entry
		update_models_csv('test.training', 'bayes_softmax3', csv_path=csv_path)

		# Add second entry
		update_models_csv('test.training', 'multi_log_regression', csv_path=csv_path)

		content = csv_path.read_text()
		lines = content.strip().split('\n')

		assert len(lines) == 3, 'Should have header + 2 entries'
		assert 'bayes_softmax3' in content
		assert 'multi_log_regression' in content

	def test_update_models_csv_duplicate(self, tmp_path):
		"""Test duplicate prevention in models.csv"""
		csv_path = tmp_path / 'models.csv'

		# Add entry
		update_models_csv('test.training', 'bayes_softmax3', csv_path=csv_path)

		# Try to add duplicate
		update_models_csv('test.training', 'bayes_softmax3', csv_path=csv_path)

		content = csv_path.read_text()
		lines = content.strip().split('\n')

		assert len(lines) == 2, 'Should have header + 1 entry (no duplicate)'


class TestEvaluateWithCrossVal:
	"""Tests for evaluate_with_cross_val function"""

	def test_evaluate_with_cross_val_signature(self):
		"""Test that evaluate_with_cross_val has correct signature"""
		import inspect

		sig = inspect.signature(evaluate_with_cross_val)
		params = list(sig.parameters.keys())
		assert 'val_base_name' in params
		assert 'model_label' in params
		assert 'prep_cfg' in params
		assert 'existing_model' in params
		assert 'n_splits' in params


class TestLoadWholeDataset:
	"""Tests for load_whole_dataset function"""

	def test_load_whole_dataset_signature(self):
		"""Test that load_whole_dataset has correct signature"""
		import inspect

		sig = inspect.signature(load_whole_dataset)
		params = list(sig.parameters.keys())
		assert 'base_name' in params
		assert 'prep_cfg' in params

	def test_train_eval_signature(self):
		"""Test that train_eval has correct signature"""
		import inspect

		sig = inspect.signature(train_eval)
		params = list(sig.parameters.keys())
		assert 'train_base' in params
		assert 'val_base' in params
		assert 'test_base' in params
		assert 'model_label' in params


class TestTrainEvalAll:
	"""Tests for train_eval_all function"""

	def test_train_eval_all_returns_dict(self):
		"""Test that train_eval_all returns dictionary of results"""
		with patch('app.model_main.train_eval') as mock_train_eval:
			mock_train_eval.return_value = {'trained': True, 'test_metrics': {'accuracy': 0.85}, 'paths': {}}

			results = train_eval_all('train', 'val', 'test')

			assert isinstance(results, dict)
			assert len(results) > 0

	def test_train_eval_all_skipped_result_structure(self):
		"""Early-exit result has the expected keys when all models already appear to exist."""
		# model_paths is fully mocked so paths['meta'].exists() returns a truthy MagicMock
		# all_exist is True, so the function returns the skipped dict immediately
		results = train_eval_all('train', 'val', 'test')
		assert results.get('status') == 'skipped'
		assert 'reason' in results
		assert results.get('train_f') == 'train'


class TestOutputLogger:
	"""Tests for OutputLogger context manager"""

	def test_captures_stdout_to_file(self, tmp_path):
		"""Content printed inside the context manager is written to the log file."""
		log_path = tmp_path / 'output.txt'
		with OutputLogger(log_path):
			print('hello from logger')
		assert log_path.exists()
		assert 'hello from logger' in log_path.read_text()

	def test_restores_stdout_after_exit(self, tmp_path):
		"""sys.stdout is restored to the original stream after the context manager exits."""
		import sys as _sys

		original_stdout = _sys.stdout
		log_path = tmp_path / 'output.txt'
		with OutputLogger(log_path):
			pass
		assert _sys.stdout is original_stdout


class TestGetOrCreateLogsDir:
	"""Tests for get_or_create_logs_dir function"""

	def test_creates_directory_if_missing(self, tmp_path):
		"""Creates the directory (including parents) when it does not exist."""
		new_dir = tmp_path / 'sub' / 'logs'
		result = get_or_create_logs_dir(new_dir)
		assert new_dir.exists()
		assert result == new_dir

	def test_returns_path_if_already_exists(self, tmp_path):
		"""Returns the path without error when the directory already exists."""
		result = get_or_create_logs_dir(tmp_path)
		assert result == tmp_path


class TestGetAppliedModelsCsvPath:
	"""Tests for get_applied_models_csv_path function"""

	def test_returns_applied_models_csv_inside_logs_dir(self, tmp_path):
		"""Path ends with 'applied_models.csv' inside the given logs_dir."""
		result = get_applied_models_csv_path(tmp_path)
		assert result.name == 'applied_models.csv'
		assert result.parent == tmp_path


class TestCheckModelAlreadyApplied:
	"""Tests for check_model_already_applied function"""

	def test_returns_none_when_no_csv(self, tmp_path):
		"""Returns None when applied_models.csv does not exist."""
		result = check_model_already_applied('model', 'type', 'data', logs_dir=tmp_path)
		assert result is None

	def test_returns_none_when_not_in_csv(self, tmp_path):
		"""Returns None when the model/type/data combination is not in the CSV."""
		register_model_application('other_model', 'bayes', 'data', 'log.txt', 'graph.png', 'cm.png', logs_dir=tmp_path)
		result = check_model_already_applied('model', 'bayes', 'data', logs_dir=tmp_path)
		assert result is None

	def test_returns_cached_result_when_found(self, tmp_path):
		"""Returns a dict with the expected keys when the entry is found."""
		register_model_application('mymodel', 'hmm_dosage', 'testdata', '/p/log.txt', '/p/graph.png', '/p/cm.png', logs_dir=tmp_path)
		result = check_model_already_applied('mymodel', 'hmm_dosage', 'testdata', logs_dir=tmp_path)
		assert result is not None
		assert 'log_file' in result
		assert 'graph_test' in result
		assert 'graph_cm' in result
		assert 'applied_date' in result


class TestRegisterModelApplication:
	"""Tests for register_model_application function"""

	def test_creates_csv_and_writes_entry(self, tmp_path):
		"""Creates applied_models.csv and writes the entry with all fields."""
		register_model_application('mymodel', 'hmm_dosage', 'testdata', 'log.txt', 'graph.png', 'cm.png', logs_dir=tmp_path)
		csv_path = tmp_path / 'applied_models.csv'
		assert csv_path.exists()
		content = csv_path.read_text()
		assert 'mymodel' in content
		assert 'hmm_dosage' in content
		assert 'testdata' in content

	def test_does_not_write_duplicate(self, tmp_path):
		"""Calling twice with the same arguments does not add a second row."""
		for _ in range(2):
			register_model_application('m', 't', 'd', 'l', 'g', 'c', logs_dir=tmp_path)
		csv_path = tmp_path / 'applied_models.csv'
		lines = csv_path.read_text().strip().split('\n')
		assert len(lines) == 2  # header + 1 entry only


class TestTestOnNewData:
	"""Tests for test_on_new_data function"""

	def test_raises_file_not_found_when_model_missing(self, tmp_path):
		"""Raises FileNotFoundError when the model meta file does not exist."""
		meta_mock = MagicMock()
		meta_mock.exists.return_value = False
		paths_mock = {'meta': meta_mock, 'idata': MagicMock(), 'dir': tmp_path}
		with patch('app.model_main.model_paths', return_value=paths_mock):
			with pytest.raises(FileNotFoundError, match='Model not found'):
				apply_to_new_data('testdata', 'hmm_dosage', 'mymodel', datasets_dir=tmp_path)
