import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

# Ensure MODELS_DIR is set so fit() checkpoint logic doesn't raise KeyError
os.environ.setdefault('MODELS_DIR', '/tmp')

# Add backend directory to path for package imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Remove any mocked model modules from sys.modules to ensure real modules are imported
# This is needed because test_main.py mocks app.model_dnn at module level
sys.modules.pop('app.model_dnn', None)
sys.modules.pop('app.model_functions', None)

# Now import the real modules
from app.model_dnn import DNNDosageClassifier, GeneticDosageNN


class TestGeneticDosageNNInitialization:
	"""Test GeneticDosageNN neural network initialization"""

	def test_init_default_parameters(self):
		"""Test neural network initialization with default parameters"""
		nn_model = GeneticDosageNN(input_dim=10)
		assert nn_model.input_dim == 10
		assert nn_model.hidden_dims == (256, 128, 64)
		assert nn_model.dropout_rate == 0.3
		assert nn_model.use_batch_norm is True
		assert nn_model.use_residual is True

	def test_init_custom_parameters(self):
		"""Test neural network initialization with custom parameters"""
		nn_model = GeneticDosageNN(input_dim=20, hidden_dims=(512, 256), dropout_rate=0.5, use_batch_norm=False, use_residual=False)
		assert nn_model.input_dim == 20
		assert nn_model.hidden_dims == (512, 256)
		assert nn_model.dropout_rate == 0.5
		assert nn_model.use_batch_norm is False
		assert nn_model.use_residual is False

	def test_forward_pass_shape(self):
		"""Test forward pass returns correct output shape"""
		nn_model = GeneticDosageNN(input_dim=10)
		X = torch.randn(16, 10)  # batch_size=16, features=10
		output = nn_model(X)
		assert output.shape == (16, 3)  # batch_size=16, 3 classes


class TestDNNDosageClassifierInitialization:
	"""Test DNNDosageClassifier initialization"""

	def test_init_default_parameters(self):
		"""Test initialization with default parameters"""
		model = DNNDosageClassifier()
		assert model.hidden_dims == (256, 128, 64)
		assert model.dropout_rate == 0.3
		assert model.learning_rate == 0.001
		assert model.batch_size == 256
		assert model.epochs == 100
		assert model.early_stopping_patience == 10
		assert model.weight_decay == 1e-4
		assert model.use_batch_norm is True
		assert model.use_residual is True
		assert isinstance(model.random_seed, int)
		# use_gpu reflects actual GPU availability (constructor does `use_gpu and GPU_AVAILABLE`)
		assert model.use_gpu == (torch.cuda.is_available() or torch.backends.mps.is_available())
		assert model.verbose is True
		assert model.use_class_weights is True

	def test_init_custom_parameters(self):
		"""Test initialization with custom parameters"""
		model = DNNDosageClassifier(
			hidden_dims=(512, 256),
			dropout_rate=0.5,
			learning_rate=0.01,
			batch_size=128,
			epochs=50,
			early_stopping_patience=5,
			weight_decay=1e-3,
			use_batch_norm=False,
			use_residual=False,
			random_seed=42,
			use_gpu=False,
			verbose=False,
			use_class_weights=False,
		)
		assert model.hidden_dims == (512, 256)
		assert model.dropout_rate == 0.5
		assert model.learning_rate == 0.01
		assert model.batch_size == 128
		assert model.epochs == 50
		assert model.early_stopping_patience == 5
		assert model.weight_decay == 1e-3
		assert model.use_batch_norm is False
		assert model.use_residual is False
		assert model.random_seed == 42
		assert model.use_gpu is False
		assert model.verbose is False
		assert model.use_class_weights is False

	def test_tag_property(self):
		"""Test model tag property"""
		model = DNNDosageClassifier()
		assert model.tag == 'dnn_dosage'

	def test_initial_state(self):
		"""Test that model state is None initially"""
		model = DNNDosageClassifier()
		assert model.model is None
		assert model.feature_mean_ is None
		assert model.feature_std_ is None
		assert model.class_weights_ is None
		assert model.pycm_train_ is None
		assert model.pycm_metrics_ is None
		assert model.train_history_ == {'loss': [], 'accuracy': []}
		assert model.val_history_ == {'loss': [], 'accuracy': []}


class TestDNNDosageClassifierFit:
	"""Test model fitting functionality"""

	def test_fit_returns_self(self):
		"""Test that fit returns self for method chaining"""
		model = DNNDosageClassifier(epochs=2, batch_size=16, verbose=False)
		X = np.random.randn(32, 10).astype(np.float32)
		y = np.array([0, 1, 2] * 10 + [0, 1], dtype=np.float32)
		groups = np.zeros(32, dtype=np.int32)

		result = model.fit(X, y, groups)
		assert result is model

	def test_fit_creates_model(self):
		"""Test that fit creates the neural network model"""
		model = DNNDosageClassifier(epochs=2, batch_size=16, verbose=False)
		X = np.random.randn(32, 10).astype(np.float32)
		y = np.array([0, 1, 2] * 10 + [0, 1], dtype=np.float32)
		groups = np.zeros(32, dtype=np.int32)

		model.fit(X, y, groups)
		assert model.model is not None
		assert isinstance(model.model, GeneticDosageNN)

	def test_fit_sets_feature_scaling(self):
		"""Test that fit sets feature scaling parameters"""
		model = DNNDosageClassifier(epochs=2, batch_size=16, verbose=False)
		X = np.random.randn(32, 10).astype(np.float32)
		y = np.array([0, 1, 2] * 10 + [0, 1], dtype=np.float32)
		groups = np.zeros(32, dtype=np.int32)

		model.fit(X, y, groups)
		assert model.feature_mean_ is not None
		assert model.feature_std_ is not None
		assert len(model.feature_mean_) == 10
		assert len(model.feature_std_) == 10

	def test_fit_creates_training_history(self):
		"""Test that fit creates training history"""
		model = DNNDosageClassifier(epochs=2, batch_size=16, verbose=False)
		X = np.random.randn(32, 10).astype(np.float32)
		y = np.array([0, 1, 2] * 10 + [0, 1], dtype=np.float32)
		groups = np.zeros(32, dtype=np.int32)

		model.fit(X, y, groups)
		assert len(model.train_history_['loss']) > 0
		assert len(model.train_history_['accuracy']) > 0
		assert len(model.val_history_['loss']) > 0
		assert len(model.val_history_['accuracy']) > 0

	def test_fit_with_class_weights(self):
		"""Test that fit computes class weights when enabled"""
		model = DNNDosageClassifier(epochs=2, batch_size=16, use_class_weights=True, verbose=False)
		X = np.random.randn(32, 10).astype(np.float32)
		y = np.array([0, 1, 2] * 10 + [0, 1], dtype=np.float32)
		groups = np.zeros(32, dtype=np.int32)

		model.fit(X, y, groups)
		assert model.class_weights_ is not None
		assert model.class_weights_.shape == (3,)


class TestDNNDosageClassifierPredict:
	"""Test prediction functionality"""

	def test_predict_proba_shape(self):
		"""Test predict_proba returns correct shape"""
		model = DNNDosageClassifier(epochs=2, batch_size=16, verbose=False)
		X_train = np.random.randn(32, 10).astype(np.float32)
		y_train = np.array([0, 1, 2] * 10 + [0, 1], dtype=np.float32)
		groups_train = np.zeros(32, dtype=np.int32)
		model.fit(X_train, y_train, groups_train)

		X_test = np.random.randn(16, 10).astype(np.float32)
		probas = model.predict_proba(X_test)

		assert probas.shape == (16, 3)
		assert np.allclose(probas.sum(axis=1), 1.0)  # Probabilities sum to 1
		assert np.all(probas >= 0) and np.all(probas <= 1)  # Valid probability range

	def test_predict_proba_without_groups(self):
		"""Test predict_proba without providing groups"""
		model = DNNDosageClassifier(epochs=2, batch_size=16, verbose=False)
		X_train = np.random.randn(32, 10).astype(np.float32)
		y_train = np.array([0, 1, 2] * 10 + [0, 1], dtype=np.float32)
		groups_train = np.zeros(32, dtype=np.int32)
		model.fit(X_train, y_train, groups_train)

		X_test = np.random.randn(16, 10).astype(np.float32)
		probas = model.predict_proba(X_test, groups=None)

		assert probas.shape == (16, 3)

	def test_predict_proba_unfitted_raises(self):
		"""Test predict_proba raises error when model not fitted"""
		model = DNNDosageClassifier()
		X_test = np.random.randn(8, 10).astype(np.float32)

		with pytest.raises(RuntimeError, match='Model must be fitted'):
			model.predict_proba(X_test)

	def test_predict_class_unfitted_raises(self):
		"""Test predict_class raises error when model not fitted"""
		model = DNNDosageClassifier()
		X_test = np.random.randn(8, 10).astype(np.float32)

		with pytest.raises(RuntimeError, match='Model must be fitted'):
			model.predict_class(X_test)

	def test_predict_class_shape(self):
		"""Test predict_class returns correct shape"""
		model = DNNDosageClassifier(epochs=2, batch_size=16, verbose=False)
		X_train = np.random.randn(32, 10).astype(np.float32)
		y_train = np.array([0, 1, 2] * 10 + [0, 1], dtype=np.float32)
		groups_train = np.zeros(32, dtype=np.int32)
		model.fit(X_train, y_train, groups_train)

		X_test = np.random.randn(16, 10).astype(np.float32)
		classes = model.predict_class(X_test)

		assert classes.shape == (16,)
		assert all(c in [0, 1, 2] for c in classes)

	def test_predict_shape(self):
		"""Test predict returns expected dosage values"""
		model = DNNDosageClassifier(epochs=2, batch_size=16, verbose=False)
		X_train = np.random.randn(32, 10).astype(np.float32)
		y_train = np.array([0, 1, 2] * 10 + [0, 1], dtype=np.float32)
		groups_train = np.zeros(32, dtype=np.int32)
		model.fit(X_train, y_train, groups_train)

		X_test = np.random.randn(16, 10).astype(np.float32)
		predictions = model.predict(X_test)

		assert predictions.shape == (16,)
		assert all(0 <= p <= 2 for p in predictions)  # Expected values in [0, 2]

	def test_predict_expected_dosage_formula(self):
		"""Test predict() equals Σ c·p(c) for c in {0,1,2}"""
		model = DNNDosageClassifier(epochs=2, batch_size=16, verbose=False)
		X_train = np.random.randn(32, 10).astype(np.float32)
		y_train = np.array([0, 1, 2] * 10 + [0, 1], dtype=np.float32)
		groups_train = np.zeros(32, dtype=np.int32)
		model.fit(X_train, y_train, groups_train)

		X_test = np.random.randn(16, 10).astype(np.float32)
		probas = model.predict_proba(X_test)
		expected_dosage = (probas * np.array([0.0, 1.0, 2.0], dtype=np.float32)).sum(axis=1)
		predictions = model.predict(X_test)

		np.testing.assert_array_almost_equal(predictions, expected_dosage)

	def test_predict_on_cpu_only(self):
		"""Test predictions work without GPU"""
		model = DNNDosageClassifier(epochs=2, batch_size=16, use_gpu=False, verbose=False)
		X_train = np.random.randn(32, 10).astype(np.float32)
		y_train = np.array([0, 1, 2] * 10 + [0, 1], dtype=np.float32)
		groups_train = np.zeros(32, dtype=np.int32)
		model.fit(X_train, y_train, groups_train)

		X_test = np.random.randn(8, 10).astype(np.float32)
		probas = model.predict_proba(X_test)

		assert probas.shape == (8, 3)


class TestDNNDosageClassifierEvaluation:
	"""Test model evaluation and metrics"""

	def test_evaluate_pycm_returns_confusion_matrix(self):
		"""Test evaluate_pycm returns ConfusionMatrix object"""
		model = DNNDosageClassifier(epochs=2, batch_size=16, verbose=False)
		X_train = np.random.randn(32, 10).astype(np.float32)
		y_train = np.array([0, 1, 2] * 10 + [0, 1], dtype=np.float32)
		groups_train = np.zeros(32, dtype=np.int32)
		model.fit(X_train, y_train, groups_train)

		X_test = np.random.randn(16, 10).astype(np.float32)
		y_test = np.array([0, 1, 2] * 5 + [0, 1], dtype=np.float32)[:16]  # Match X_test length

		from pycm import ConfusionMatrix

		pycm = model.evaluate_pycm(X_test, y_test)
		assert isinstance(pycm, ConfusionMatrix)

	def test_pycm_metrics_populated_after_fit(self):
		"""Test that PYCM metrics are populated after training"""
		model = DNNDosageClassifier(epochs=2, batch_size=16, verbose=False)
		X_train = np.random.randn(32, 10).astype(np.float32)
		y_train = np.array([0, 1, 2] * 10 + [0, 1], dtype=np.float32)
		groups_train = np.zeros(32, dtype=np.int32)

		model.fit(X_train, y_train, groups_train)
		assert model.pycm_metrics_ is not None
		assert 'overall_accuracy' in model.pycm_metrics_
		assert 'kappa' in model.pycm_metrics_

	def test_pycm_metrics_all_keys_present(self):
		"""Test that all expected keys are present in pycm_metrics_ after fit"""
		model = DNNDosageClassifier(epochs=2, batch_size=16, verbose=False)
		X_train = np.random.randn(32, 10).astype(np.float32)
		y_train = np.array([0, 1, 2] * 10 + [0, 1], dtype=np.float32)
		groups_train = np.zeros(32, dtype=np.int32)
		model.fit(X_train, y_train, groups_train)

		required_keys = {
			'overall_accuracy',
			'kappa',
			'overall_f1',
			'overall_precision',
			'overall_recall',
			'class_accuracy',
			'class_f1',
			'class_precision',
			'class_recall',
		}
		assert required_keys <= set(model.pycm_metrics_.keys())


class TestDNNDosageClassifierPersistence:
	"""Test model saving and loading"""

	def test_save_creates_files(self):
		"""Test that save creates necessary files"""
		model = DNNDosageClassifier(epochs=2, batch_size=16, verbose=False)
		X = np.random.randn(32, 10).astype(np.float32)
		y = np.array([0, 1, 2] * 10 + [0, 1], dtype=np.float32)
		groups = np.zeros(32, dtype=np.int32)
		model.fit(X, y, groups)

		with tempfile.TemporaryDirectory() as tmpdir:
			tmpdir_path = Path(tmpdir)
			paths = {'dir': tmpdir_path, 'meta': tmpdir_path / 'meta.json'}
			extra_meta = {'dataset': 'test', 'split': 'training'}

			model.save(paths, extra_meta)

			assert paths['meta'].exists()
			# Check for model file
			model_files = list(tmpdir_path.glob('*.model.pt'))
			assert len(model_files) > 0

	def test_save_unfitted_raises(self):
		"""Test that save raises error when model not fitted"""
		model = DNNDosageClassifier()

		with tempfile.TemporaryDirectory() as tmpdir:
			tmpdir_path = Path(tmpdir)
			paths = {'dir': tmpdir_path, 'meta': tmpdir_path / 'meta.json'}
			extra_meta = {}

			with pytest.raises(RuntimeError, match='No model to save'):
				model.save(paths, extra_meta)

	def test_load_restores_feature_scaling(self):
		"""Test that load restores feature scaling parameters"""
		model = DNNDosageClassifier(epochs=2, batch_size=16, verbose=False)
		X = np.random.randn(32, 10).astype(np.float32)
		y = np.array([0, 1, 2] * 10 + [0, 1], dtype=np.float32)
		groups = np.zeros(32, dtype=np.int32)
		model.fit(X, y, groups)

		with tempfile.TemporaryDirectory() as tmpdir:
			tmpdir_path = Path(tmpdir)
			paths = {'dir': tmpdir_path, 'meta': tmpdir_path / 'meta.json'}
			extra_meta = {'dataset': 'test'}

			model.save(paths, extra_meta)

			# Load the model
			loaded_model = DNNDosageClassifier.load(paths)

			assert loaded_model.feature_mean_ is not None
			assert loaded_model.feature_std_ is not None
			assert np.allclose(loaded_model.feature_mean_, model.feature_mean_)
			assert np.allclose(loaded_model.feature_std_, model.feature_std_)

	def test_load_preserves_hyperparameters(self):
		"""Test that load preserves model hyperparameters"""
		custom_params = {'hidden_dims': (512, 256), 'dropout_rate': 0.4, 'learning_rate': 0.005, 'epochs': 50, 'use_gpu': False}
		model = DNNDosageClassifier(**custom_params, batch_size=16, verbose=False)
		X = np.random.randn(32, 10).astype(np.float32)
		y = np.array([0, 1, 2] * 10 + [0, 1], dtype=np.float32)
		groups = np.zeros(32, dtype=np.int32)
		model.fit(X, y, groups)

		with tempfile.TemporaryDirectory() as tmpdir:
			tmpdir_path = Path(tmpdir)
			paths = {'dir': tmpdir_path, 'meta': tmpdir_path / 'meta.json'}
			extra_meta = {}

			model.save(paths, extra_meta)
			loaded_model = DNNDosageClassifier.load(paths)

			assert loaded_model.hidden_dims == custom_params['hidden_dims']
			assert loaded_model.dropout_rate == custom_params['dropout_rate']
			assert loaded_model.learning_rate == custom_params['learning_rate']
			assert loaded_model.epochs == custom_params['epochs']

	def test_save_load_predict_roundtrip(self):
		"""Test that predictions are identical before and after save/load"""
		model = DNNDosageClassifier(epochs=2, batch_size=16, use_gpu=False, verbose=False)
		X_train = np.random.randn(32, 10).astype(np.float32)
		y_train = np.array([0, 1, 2] * 10 + [0, 1], dtype=np.float32)
		groups_train = np.zeros(32, dtype=np.int32)
		model.fit(X_train, y_train, groups_train)

		X_test = np.random.randn(8, 10).astype(np.float32)
		proba_before = model.predict_proba(X_test)
		classes_before = model.predict_class(X_test)

		with tempfile.TemporaryDirectory() as tmpdir:
			tmpdir_path = Path(tmpdir)
			paths = {'dir': tmpdir_path, 'meta': tmpdir_path / 'meta.json'}
			model.save(paths, {})
			loaded = DNNDosageClassifier.load(paths)

		proba_after = loaded.predict_proba(X_test)
		classes_after = loaded.predict_class(X_test)

		np.testing.assert_allclose(proba_after, proba_before, rtol=1e-5)
		np.testing.assert_array_equal(classes_after, classes_before)


class TestDNNDosageClassifierEdgeCases:
	"""Test edge cases and error handling"""

	def test_predict_with_single_sample(self):
		"""Test prediction with single sample"""
		model = DNNDosageClassifier(epochs=2, batch_size=16, verbose=False)
		X_train = np.random.randn(32, 10).astype(np.float32)
		y_train = np.array([0, 1, 2] * 10 + [0, 1], dtype=np.float32)
		groups_train = np.zeros(32, dtype=np.int32)
		model.fit(X_train, y_train, groups_train)

		X_single = np.random.randn(1, 10).astype(np.float32)
		proba = model.predict_proba(X_single)

		assert proba.shape == (1, 3)

	def test_predict_proba_large_batch(self):
		"""Test prediction on larger batch"""
		model = DNNDosageClassifier(epochs=2, batch_size=16, verbose=False)
		X_train = np.random.randn(32, 10).astype(np.float32)
		y_train = np.array([0, 1, 2] * 10 + [0, 1], dtype=np.float32)
		groups_train = np.zeros(32, dtype=np.int32)
		model.fit(X_train, y_train, groups_train)

		X_large = np.random.randn(2048, 10).astype(np.float32)
		proba = model.predict_proba(X_large)

		assert proba.shape == (2048, 3)

	def test_imbalanced_classes(self):
		"""Test with imbalanced class distribution"""
		model = DNNDosageClassifier(epochs=2, batch_size=16, use_class_weights=True, verbose=False)
		X = np.random.randn(32, 10).astype(np.float32)
		# Heavily imbalanced: mostly class 0
		y = np.array([0] * 20 + [1] * 10 + [2] * 2, dtype=np.float32)
		groups = np.zeros(32, dtype=np.int32)

		result = model.fit(X, y, groups)
		assert result is model
		assert model.class_weights_ is not None
