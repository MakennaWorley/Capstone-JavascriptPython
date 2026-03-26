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
	_select_model,
	check_gpu_status,
	evaluate_with_cross_val,
	get_optimal_training_config,
	load_whole_dataset,
	train_eval,
	train_eval_all,
	update_models_csv,
)


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
		ModelCls, tag = _select_model('bayes_softmax3')
		assert tag == 'bayes_softmax3'
		assert ModelCls is not None

	def test_select_model_sklearn(self):
		"""Test model selection - Sklearn"""
		ModelCls, tag = _select_model('multi_log_regression')
		assert tag == 'multi_log_regression'
		assert ModelCls is not None

	def test_select_model_hmm(self):
		"""Test model selection - HMM"""
		ModelCls, tag = _select_model('hmm_dosage')
		assert tag == 'hmm_dosage'
		assert ModelCls is not None

	def test_select_model_dnn(self):
		"""Test model selection - DNN"""
		ModelCls, tag = _select_model('dnn_dosage')
		assert tag == 'dnn_dosage'
		assert ModelCls is not None

	def test_select_model_gnn(self):
		"""Test model selection - GNN"""
		ModelCls, tag = _select_model('gnn_dosage')
		assert tag == 'gnn_dosage'
		assert ModelCls is not None

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
