import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Add backend directory to path for package imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Remove any mocked model modules from sys.modules
sys.modules.pop('app.model_graph_functions', None)

# Now import the real module
from app.model_graph_functions import evaluate_and_graph_clf, evaluate_and_graph_reg, plot_confusion_matrix


class TestEvaluateAndGraphClf:
	"""Test evaluate_and_graph_clf function for classification"""

	def test_evaluate_clf_basic_metrics(self, capsys):
		"""Test basic metric calculation for classifier"""
		# Create a mock model
		model = MagicMock()
		model.predict_class = MagicMock(return_value=np.array([0, 1, 2, 0, 1]))
		model.predict_proba = MagicMock(
			return_value=np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.85, 0.1, 0.05], [0.1, 0.85, 0.05]])
		)

		X = np.random.randn(5, 10).astype(np.float32)
		y = np.array([0, 1, 2, 0, 1], dtype=np.float32)

		result = evaluate_and_graph_clf(model, X, y, 'test_model', graph=False)

		assert result['model'] == 'test_model'
		assert 'accuracy' in result
		assert 'balanced_accuracy' in result
		assert 'f1_macro' in result
		assert 'f1_weighted' in result
		assert 'auc_macro' in result
		assert 0 <= result['accuracy'] <= 1
		assert 0 <= result['balanced_accuracy'] <= 1

	def test_evaluate_clf_perfect_predictions(self, capsys):
		"""Test metrics when predictions are perfect"""
		model = MagicMock()
		model.predict_class = MagicMock(return_value=np.array([0, 1, 2, 0, 1]))
		model.predict_proba = MagicMock(return_value=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))

		X = np.random.randn(5, 10).astype(np.float32)
		y = np.array([0, 1, 2, 0, 1], dtype=np.float32)

		result = evaluate_and_graph_clf(model, X, y, 'perfect_model', graph=False)

		assert result['accuracy'] == 1.0
		assert result['balanced_accuracy'] == 1.0
		assert result['f1_macro'] == 1.0

	def test_evaluate_clf_worst_predictions(self, capsys):
		"""Test metrics when predictions are completely wrong"""
		model = MagicMock()
		model.predict_class = MagicMock(return_value=np.array([2, 2, 2, 2, 2]))
		model.predict_proba = MagicMock(return_value=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]))

		X = np.random.randn(5, 10).astype(np.float32)
		y = np.array([0, 1, 2, 0, 1], dtype=np.float32)

		result = evaluate_and_graph_clf(model, X, y, 'worst_model', graph=False)

		assert result['accuracy'] < 0.5

	def test_evaluate_clf_prints_metrics(self, capsys):
		"""Test that metrics are printed"""
		model = MagicMock()
		model.predict_class = MagicMock(return_value=np.array([0, 1, 2]))
		model.predict_proba = MagicMock(return_value=np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]))

		X = np.random.randn(3, 10).astype(np.float32)
		y = np.array([0, 1, 2], dtype=np.float32)

		evaluate_and_graph_clf(model, X, y, 'test_model', graph=False)
		captured = capsys.readouterr()

		assert 'test_model' in captured.out
		assert 'Accuracy' in captured.out
		assert 'F1-Score' in captured.out

	@patch('matplotlib.pyplot.tight_layout')
	def test_evaluate_clf_with_graph(self, mock_tight_layout):
		"""Test that graphing creates plots when graph=True"""
		model = MagicMock()
		model.predict_class = MagicMock(return_value=np.array([0, 1, 2, 0, 1, 2]))
		model.predict_proba = MagicMock(
			return_value=np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.85, 0.1, 0.05], [0.15, 0.75, 0.1], [0.05, 0.05, 0.9]])
		)

		X = np.random.randn(6, 10).astype(np.float32)
		y = np.array([0, 1, 2, 0, 1, 2], dtype=np.float32)

		result = evaluate_and_graph_clf(model, X, y, 'test_model', graph=True)

		assert result['model'] == 'test_model'
		mock_tight_layout.assert_called()

	def test_evaluate_clf_with_kwargs(self):
		"""Test that kwargs are passed to model methods"""
		model = MagicMock()
		model.predict_class = MagicMock(return_value=np.array([0, 1, 2]))
		model.predict_proba = MagicMock(return_value=np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]))

		X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
		y = np.array([0, 1, 2], dtype=np.float32)

		evaluate_and_graph_clf(model, X, y, 'test', graph=False, groups=np.array([0, 0, 0]))

		# Verify the method was called with the expected arguments
		model.predict_class.assert_called_once()
		call_args = model.predict_class.call_args
		np.testing.assert_array_equal(call_args[0][0], X)
		np.testing.assert_array_equal(call_args[1]['groups'], np.array([0, 0, 0]))

	def test_evaluate_clf_fallback_no_predict_class(self):
		"""Test fallback to model.predict() when predict_class is absent"""
		model = MagicMock(spec=['predict', 'predict_proba'])  # no predict_class
		model.predict = MagicMock(return_value=np.array([0, 1, 2]))
		model.predict_proba = MagicMock(return_value=np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]))

		X = np.random.randn(3, 10).astype(np.float32)
		y = np.array([0, 1, 2], dtype=np.float32)

		result = evaluate_and_graph_clf(model, X, y, 'fallback_model', graph=False)

		model.predict.assert_called_once()
		assert result['accuracy'] == 1.0

	def test_evaluate_clf_fallback_no_predict_proba(self):
		"""Test fallback to label_binarize when predict_proba is absent"""
		model = MagicMock(spec=['predict_class'])  # no predict_proba
		model.predict_class = MagicMock(return_value=np.array([0, 1, 2]))

		X = np.random.randn(3, 10).astype(np.float32)
		y = np.array([0, 1, 2], dtype=np.float32)

		result = evaluate_and_graph_clf(model, X, y, 'no_proba_model', graph=False)

		assert 'accuracy' in result
		assert 'auc_macro' in result


class TestEvaluateAndGraphReg:
	"""Test evaluate_and_graph_reg function for regression"""

	def test_evaluate_reg_basic_metrics(self, capsys):
		"""Test basic metric calculation for regressor"""
		model = MagicMock()
		model.predict = MagicMock(return_value=np.array([0.9, 1.1, 1.95, 0.05, 2.0]))

		X = np.random.randn(5, 10).astype(np.float32)
		y = np.array([1.0, 1.0, 2.0, 0.0, 2.0], dtype=np.float32)

		result = evaluate_and_graph_reg(model, X, y, 'test_model', graph=False)

		assert result['model'] == 'test_model'
		assert 'rmse' in result
		assert 'r2' in result
		assert result['rmse'] >= 0
		assert -1 <= result['r2'] <= 1

	def test_evaluate_reg_perfect_predictions(self, capsys):
		"""Test metrics when predictions are perfect"""
		model = MagicMock()
		y_true = np.array([0.0, 1.0, 2.0, 1.5, 0.5], dtype=np.float32)
		model.predict = MagicMock(return_value=y_true.copy())

		X = np.random.randn(5, 10).astype(np.float32)

		result = evaluate_and_graph_reg(model, X, y_true, 'perfect_model', graph=False)

		assert np.isclose(result['rmse'], 0.0, atol=1e-6)
		assert np.isclose(result['r2'], 1.0, atol=1e-6)

	def test_evaluate_reg_prints_metrics(self, capsys):
		"""Test that metrics are printed"""
		model = MagicMock()
		model.predict = MagicMock(return_value=np.array([1.0, 2.0, 3.0]))

		X = np.random.randn(3, 10).astype(np.float32)
		y = np.array([1.0, 2.0, 3.0], dtype=np.float32)

		evaluate_and_graph_reg(model, X, y, 'test_model', graph=False)
		captured = capsys.readouterr()

		assert 'test_model' in captured.out
		assert 'RMSE' in captured.out
		assert 'R²' in captured.out

	@patch('matplotlib.pyplot.tight_layout')
	def test_evaluate_reg_with_graph(self, mock_tight_layout):
		"""Test that graphing creates plots when graph=True"""
		model = MagicMock()
		y_true = np.array([0.0, 1.0, 1.5, 2.0, 0.5, 1.5, 2.0], dtype=np.float32)
		y_pred = np.array([0.1, 0.9, 1.6, 1.95, 0.4, 1.65, 1.9], dtype=np.float32)
		model.predict = MagicMock(return_value=y_pred)

		X = np.random.randn(7, 10).astype(np.float32)

		result = evaluate_and_graph_reg(model, X, y_true, 'test_model', graph=True)

		assert result['model'] == 'test_model'
		mock_tight_layout.assert_called()

	def test_evaluate_reg_with_nan_values(self, capsys):
		"""Test handling of NaN values"""
		model = MagicMock()
		model.predict = MagicMock(return_value=np.array([0.9, np.nan, 2.0, 0.0, 2.0]))

		X = np.random.randn(5, 10).astype(np.float32)
		y = np.array([1.0, 1.0, 2.0, 0.0, 2.0], dtype=np.float32)

		# Should not raise error, filters out NaN
		result = evaluate_and_graph_reg(model, X, y, 'test_model', graph=False)
		assert 'rmse' in result

	def test_evaluate_reg_with_inf_values(self, capsys):
		"""Test handling of infinite values"""
		model = MagicMock()
		model.predict = MagicMock(return_value=np.array([0.9, np.inf, 2.0, 0.0, 2.0]))

		X = np.random.randn(5, 10).astype(np.float32)
		y = np.array([1.0, 1.0, 2.0, 0.0, 2.0], dtype=np.float32)

		# Should not raise error, filters out inf
		result = evaluate_and_graph_reg(model, X, y, 'test_model', graph=False)
		assert 'rmse' in result

	def test_evaluate_reg_no_valid_values_raises(self):
		"""Test error when all values are invalid"""
		model = MagicMock()
		model.predict = MagicMock(return_value=np.array([np.nan, np.nan, np.inf]))

		X = np.random.randn(3, 10).astype(np.float32)
		y = np.array([np.nan, np.nan, np.inf], dtype=np.float32)

		with pytest.raises(ValueError, match='No finite.*values'):
			evaluate_and_graph_reg(model, X, y, 'test_model', graph=False)

	def test_evaluate_reg_with_kwargs(self):
		"""Test that kwargs are passed to model methods"""
		model = MagicMock()
		model.predict = MagicMock(return_value=np.array([1.0, 2.0]))

		X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
		y = np.array([1.0, 2.0], dtype=np.float32)

		evaluate_and_graph_reg(model, X, y, 'test', graph=False, groups=np.array([0, 0]))

		# Verify the method was called with the expected arguments
		model.predict.assert_called_once()
		call_args = model.predict.call_args
		np.testing.assert_array_equal(call_args[0][0], X)
		np.testing.assert_array_equal(call_args[1]['groups'], np.array([0, 0]))


class TestPlotConfusionMatrix:
	"""Test plot_confusion_matrix function"""

	def test_confusion_matrix_basic(self):
		"""Test basic confusion matrix creation"""
		y_true = np.array([0, 1, 2, 0, 1, 2])
		y_pred = np.array([0, 1, 2, 0, 1, 2])

		with tempfile.TemporaryDirectory() as tmpdir:
			save_path = Path(tmpdir) / 'cm.png'
			plot_confusion_matrix(y_true, y_pred, 'test', str(save_path))

			assert save_path.exists()

	def test_confusion_matrix_with_errors(self):
		"""Test confusion matrix with some misclassifications"""
		y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
		y_pred = np.array([0, 1, 1, 0, 2, 2, 1, 1, 2])

		with tempfile.TemporaryDirectory() as tmpdir:
			save_path = Path(tmpdir) / 'cm.png'
			plot_confusion_matrix(y_true, y_pred, 'test_errors', str(save_path))

			assert save_path.exists()

	def test_confusion_matrix_float_input(self):
		"""Test that float inputs are rounded correctly"""
		y_true = np.array([0.1, 0.9, 2.1, 0.2, 1.1, 1.9])
		y_pred = np.array([0.0, 1.2, 2.0, 0.3, 1.0, 2.1])

		with tempfile.TemporaryDirectory() as tmpdir:
			save_path = Path(tmpdir) / 'cm_float.png'
			plot_confusion_matrix(y_true, y_pred, 'float_test', str(save_path))

			assert save_path.exists()

	def test_confusion_matrix_saves_to_directory(self):
		"""Test that file is saved to specified directory"""
		y_true = np.array([0, 1, 2])
		y_pred = np.array([0, 1, 2])

		with tempfile.TemporaryDirectory() as tmpdir:
			# Create parent directory first (function doesn't create it)
			subdir = Path(tmpdir) / 'subdir'
			subdir.mkdir(parents=True, exist_ok=True)
			save_path = subdir / 'cm.png'

			plot_confusion_matrix(y_true, y_pred, 'test', str(save_path))

			# File should be created
			assert save_path.exists()

	def test_confusion_matrix_file_format(self):
		"""Test that output is a PNG file"""
		y_true = np.array([0, 1, 2, 0, 1])
		y_pred = np.array([0, 1, 2, 0, 1])

		with tempfile.TemporaryDirectory() as tmpdir:
			save_path = Path(tmpdir) / 'cm.png'
			plot_confusion_matrix(y_true, y_pred, 'test', str(save_path))

			# Check that it's a valid PNG
			with open(save_path, 'rb') as f:
				header = f.read(8)
				# PNG file signature
				assert header == b'\x89PNG\r\n\x1a\n'

	def test_confusion_matrix_perfect_predictions(self):
		"""Test confusion matrix with perfect predictions"""
		y_true = np.array([0, 1, 2, 0, 1, 2])
		y_pred = np.array([0, 1, 2, 0, 1, 2])

		with tempfile.TemporaryDirectory() as tmpdir:
			save_path = Path(tmpdir) / 'cm_perfect.png'
			plot_confusion_matrix(y_true, y_pred, 'perfect', str(save_path))

			assert save_path.exists()
			# File should have some size (not empty)
			assert save_path.stat().st_size > 0

	def test_confusion_matrix_single_class(self):
		"""Test confusion matrix with single class present"""
		y_true = np.array([0, 0, 0, 0])
		y_pred = np.array([0, 0, 0, 0])

		with tempfile.TemporaryDirectory() as tmpdir:
			save_path = Path(tmpdir) / 'cm_single.png'
			plot_confusion_matrix(y_true, y_pred, 'single_class', str(save_path))

			assert save_path.exists()


class TestGraphFunctionsIntegration:
	"""Integration tests combining multiple functions"""

	def test_clf_and_reg_both_work(self):
		"""Test that both classification and regression evaluation work together"""
		# Classification
		clf_model = MagicMock()
		clf_model.predict_class = MagicMock(return_value=np.array([0, 1, 2]))
		clf_model.predict_proba = MagicMock(return_value=np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]))

		X = np.random.randn(3, 10).astype(np.float32)
		y = np.array([0, 1, 2], dtype=np.float32)

		clf_result = evaluate_and_graph_clf(clf_model, X, y, 'clf', graph=False)
		assert clf_result['accuracy'] > 0

		# Regression
		reg_model = MagicMock()
		reg_model.predict = MagicMock(return_value=np.array([0.1, 1.0, 2.0]))

		reg_result = evaluate_and_graph_reg(reg_model, X, y, 'reg', graph=False)
		assert reg_result['rmse'] >= 0

	def test_with_real_arrays(self):
		"""Test with real data arrays similar to actual use"""
		np.random.seed(42)
		n_samples = 100
		n_features = 20

		X = np.random.randn(n_samples, n_features).astype(np.float32)
		y = np.tile([0, 1, 2], n_samples // 3 + 1)[:n_samples]

		# Classification
		clf_model = MagicMock()
		y_pred_clf = np.random.choice([0, 1, 2], n_samples)
		clf_model.predict_class = MagicMock(return_value=y_pred_clf)
		clf_model.predict_proba = MagicMock(return_value=np.random.dirichlet([1, 1, 1], n_samples))

		result = evaluate_and_graph_clf(clf_model, X, y, 'random_clf', graph=False)
		assert 0 <= result['accuracy'] <= 1
		assert 0 <= result['balanced_accuracy'] <= 1

		# Regression
		reg_model = MagicMock()
		y_pred_reg = y.astype(float) + np.random.normal(0, 0.3, n_samples)
		reg_model.predict = MagicMock(return_value=y_pred_reg)

		result = evaluate_and_graph_reg(reg_model, X, y, 'random_reg', graph=False)
		assert result['rmse'] >= 0
