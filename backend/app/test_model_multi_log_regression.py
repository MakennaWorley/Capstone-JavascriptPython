"""
Comprehensive pytest suite for SklearnMultinomialClassifier model.

Tests cover initialization, fitting, predictions, persistence, and edge cases.
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Handle module cleanup for proper imports
if 'app.model_multi_log_regression' in sys.modules:
	del sys.modules['app.model_multi_log_regression']
if 'model_multi_log_regression' in sys.modules:
	del sys.modules['model_multi_log_regression']

from model_multi_log_regression import SklearnMultinomialClassifier


class TestSklearnMultinomialClassifierInitialization:
	"""Test model initialization and properties."""

	def test_init_default_parameters(self):
		"""Test initialization with default parameters."""
		model = SklearnMultinomialClassifier()
		assert model.model is not None
		assert model.scaler is not None
		assert model.tag == 'multi_log_regression'

	def test_init_custom_random_seed(self):
		"""Test initialization with custom random seed."""
		model = SklearnMultinomialClassifier(random_seed=42)
		assert model.model.random_state == 42

	def test_tag_property(self):
		"""Test tag property returns correct string."""
		model = SklearnMultinomialClassifier()
		assert model.tag == 'multi_log_regression'
		assert isinstance(model.tag, str)

	def test_model_not_fitted_initially(self):
		"""Test that model is not fitted upon initialization."""
		model = SklearnMultinomialClassifier()
		assert not hasattr(model.model, 'coef_')
		assert not hasattr(model.scaler, 'mean_')

	def test_init_creates_new_instance(self):
		"""Test that each initialization creates independent instances."""
		model1 = SklearnMultinomialClassifier(random_seed=42)
		model2 = SklearnMultinomialClassifier(random_seed=42)
		assert model1.model is not model2.model
		assert model1.scaler is not model2.scaler


class TestSklearnMultinomialClassifierFitting:
	"""Test model fitting functionality."""

	@pytest.fixture
	def sample_data(self):
		"""Generate sample data for testing."""
		np.random.seed(42)
		X = np.random.randn(100, 10).astype(np.float32)
		y = np.tile([0, 1, 2], 34)[:100]  # 100 samples with dosages
		return X, y

	@pytest.fixture
	def model(self):
		"""Create a fresh model instance."""
		return SklearnMultinomialClassifier(random_seed=42)

	def test_fit_basic(self, model, sample_data):
		"""Test basic fitting."""
		X, y = sample_data
		result = model.fit(X, y)
		assert result is model  # Check that fit returns self
		assert hasattr(model.model, 'coef_')
		assert hasattr(model.scaler, 'mean_')

	def test_fit_scales_features(self, model, sample_data):
		"""Test that fitting properly scales features."""
		X, y = sample_data
		model.fit(X, y)

		# Check scaler is fitted
		assert hasattr(model.scaler, 'mean_')
		assert hasattr(model.scaler, 'scale_')
		assert model.scaler.mean_ is not None
		assert model.scaler.scale_ is not None

	def test_fit_sets_model_classes(self, model, sample_data):
		"""Test that model learns the dosage classes."""
		X, y = sample_data
		model.fit(X, y)

		# sklearn sets classes_ after fit
		assert hasattr(model.model, 'classes_')
		assert len(model.model.classes_) == 3

	def test_fit_with_coerce_dosage(self, model):
		"""Test that fit coerces dosage to integers."""
		X = np.random.randn(30, 5).astype(np.float32)
		y = np.array([0.1, 0.8, 2.1, 0.9, 1.5] * 6)  # Float dosages

		model.fit(X, y)
		assert hasattr(model.model, 'coef_')

	def test_fit_different_feature_sizes(self, model):
		"""Test fitting with different feature dimensions."""
		np.random.seed(42)
		for n_features in [3, 5, 10, 20]:
			X = np.random.randn(50, n_features).astype(np.float32)
			y = np.tile([0, 1, 2], 17)[:50]

			model_tmp = SklearnMultinomialClassifier(random_seed=42)
			model_tmp.fit(X, y)
			assert model_tmp.model.n_features_in_ == n_features

	def test_fit_with_groups_ignored(self, model, sample_data):
		"""Test that groups parameter is accepted but not used."""
		X, y = sample_data
		groups = np.random.randint(0, 3, len(y))

		# Should not raise error
		model.fit(X, y, groups=groups)
		assert hasattr(model.model, 'coef_')

	def test_fit_coerces_y_int(self, model):
		"""Test that fit converts y to integers."""
		X = np.random.randn(30, 5).astype(np.float32)
		y_float = np.array([0.4, 1.6, 2.2, 0.8] * 8)[:30]

		model.fit(X, y_float)
		# Model should have learned from integers (0 or 2)
		assert hasattr(model.model, 'coef_')


class TestSklearnMultinomialClassifierPrediction:
	"""Test prediction methods."""

	@pytest.fixture
	def fitted_model(self):
		"""Create and fit a model."""
		model = SklearnMultinomialClassifier(random_seed=42)
		np.random.seed(42)
		X_train = np.random.randn(100, 8).astype(np.float32)
		y_train = np.tile([0, 1, 2], 34)[:100]
		model.fit(X_train, y_train)
		return model

	def test_predict_proba_shape(self, fitted_model):
		"""Test predict_proba returns correct shape."""
		X = np.random.randn(20, 8).astype(np.float32)
		proba = fitted_model.predict_proba(X)

		assert proba.shape == (20, 3)
		assert proba.dtype == np.float32

	def test_predict_proba_sums_to_one(self, fitted_model):
		"""Test that probabilities sum to 1."""
		X = np.random.randn(20, 8).astype(np.float32)
		proba = fitted_model.predict_proba(X)

		sums = proba.sum(axis=1)
		np.testing.assert_array_almost_equal(sums, np.ones(20))

	def test_predict_proba_in_valid_range(self, fitted_model):
		"""Test that probabilities are in [0, 1]."""
		X = np.random.randn(20, 8).astype(np.float32)
		proba = fitted_model.predict_proba(X)

		assert np.all(proba >= 0)
		assert np.all(proba <= 1)

	def test_predict_class_shape(self, fitted_model):
		"""Test predict_class returns 1D array."""
		X = np.random.randn(20, 8).astype(np.float32)
		predictions = fitted_model.predict_class(X)

		assert predictions.shape == (20,)

	def test_predict_class_valid_dosages(self, fitted_model):
		"""Test that predictions are valid dosages {0, 1, 2}."""
		X = np.random.randn(20, 8).astype(np.float32)
		predictions = fitted_model.predict_class(X)

		assert np.all(np.isin(predictions, [0, 1, 2]))

	def test_predict_returns_expected_value(self, fitted_model):
		"""Test predict returns expected dosage."""
		X = np.random.randn(20, 8).astype(np.float32)
		predictions = fitted_model.predict(X)

		assert predictions.shape == (20,)
		assert np.all(predictions >= 0)
		assert np.all(predictions <= 2)

	def test_predict_matches_expected_value_formula(self, fitted_model):
		"""Test predict matches E[y] = sum(p_i * i)."""
		X = np.random.randn(20, 8).astype(np.float32)
		proba = fitted_model.predict_proba(X)
		predictions = fitted_model.predict(X)

		expected = (proba * np.array([0.0, 1.0, 2.0])[None, :]).sum(axis=1)
		np.testing.assert_array_almost_equal(predictions, expected)

	def test_predict_proba_unfitted_raises_error(self):
		"""Test that predict_proba raises error on unfitted model."""
		model = SklearnMultinomialClassifier()
		X = np.random.randn(10, 5).astype(np.float32)

		with pytest.raises(RuntimeError, match='must be fitted'):
			model.predict_proba(X)

	def test_predict_class_unfitted_raises_error(self):
		"""Test that predict_class raises error on unfitted model."""
		model = SklearnMultinomialClassifier()
		X = np.random.randn(10, 5).astype(np.float32)

		with pytest.raises(RuntimeError, match='must be fitted'):
			model.predict_class(X)

	def test_predict_with_groups_ignored(self, fitted_model):
		"""Test that groups parameter is accepted."""
		X = np.random.randn(20, 8).astype(np.float32)
		groups = np.random.randint(0, 3, 20)

		# Should not raise error
		predictions = fitted_model.predict(X, groups=groups)
		assert predictions.shape == (20,)

	def test_predict_single_sample(self, fitted_model):
		"""Test prediction on single sample."""
		X = np.random.randn(1, 8).astype(np.float32)

		proba = fitted_model.predict_proba(X)
		cls = fitted_model.predict_class(X)
		pred = fitted_model.predict(X)

		assert proba.shape == (1, 3)
		assert cls.shape == (1,)
		assert pred.shape == (1,)

	def test_predict_multiple_batches(self, fitted_model):
		"""Test predictions on multiple batches."""
		predictions_list = []
		for _ in range(3):
			X = np.random.randn(10, 8).astype(np.float32)
			pred = fitted_model.predict(X)
			predictions_list.append(pred)

		# All predictions should be valid
		for pred in predictions_list:
			assert np.all(np.isin(np.rint(pred).astype(int), [0, 1, 2]))


class TestSklearnMultinomialClassifierPersistence:
	"""Test model saving and loading."""

	@pytest.fixture
	def fitted_model(self):
		"""Create and fit a model."""
		model = SklearnMultinomialClassifier(random_seed=42)
		np.random.seed(42)
		X_train = np.random.randn(100, 8).astype(np.float32)
		y_train = np.tile([0, 1, 2], 34)[:100]
		model.fit(X_train, y_train)
		return model

	def test_save_basic(self, fitted_model):
		"""Test basic save functionality."""
		with tempfile.TemporaryDirectory() as tmpdir:
			model_dir = Path(tmpdir) / 'models'
			model_dir.mkdir(parents=True)
			meta_path = model_dir / 'test_model.json'

			paths = {'dir': model_dir, 'meta': meta_path}
			fitted_model.save(paths, extra_meta={'version': '1.0'})

			assert meta_path.exists()
			assert meta_path.stat().st_size > 0

	def test_save_creates_valid_json(self, fitted_model):
		"""Test that save creates valid JSON."""
		with tempfile.TemporaryDirectory() as tmpdir:
			model_dir = Path(tmpdir) / 'models'
			model_dir.mkdir(parents=True)
			meta_path = model_dir / 'test_model.json'

			paths = {'dir': model_dir, 'meta': meta_path}
			fitted_model.save(paths, extra_meta={'version': '1.0'})

			with open(meta_path) as f:
				data = json.load(f)

			assert data['type'] == 'SklearnMultinomialClassifier'
			assert data['tag'] == 'multi_log_regression'

	def test_save_preserves_parameters(self, fitted_model):
		"""Test that save preserves model parameters."""
		with tempfile.TemporaryDirectory() as tmpdir:
			model_dir = Path(tmpdir) / 'models'
			model_dir.mkdir(parents=True)
			meta_path = model_dir / 'test_model.json'

			paths = {'dir': model_dir, 'meta': meta_path}
			fitted_model.save(paths, extra_meta={'version': '1.0'})

			with open(meta_path) as f:
				data = json.load(f)

			# Check that all required parameters are present
			assert 'feature_mean' in data
			assert 'feature_std' in data
			assert 'coef' in data
			assert 'intercept' in data

	def test_save_unfitted_raises_error(self):
		"""Test that save raises error on unfitted model."""
		model = SklearnMultinomialClassifier()

		with tempfile.TemporaryDirectory() as tmpdir:
			model_dir = Path(tmpdir) / 'models'
			model_dir.mkdir(parents=True)
			meta_path = model_dir / 'test_model.json'

			paths = {'dir': model_dir, 'meta': meta_path}
			with pytest.raises(RuntimeError, match='must be fitted'):
				model.save(paths, extra_meta={})

	def test_load_basic(self, fitted_model):
		"""Test basic load functionality."""
		with tempfile.TemporaryDirectory() as tmpdir:
			model_dir = Path(tmpdir) / 'models'
			model_dir.mkdir(parents=True)
			meta_path = model_dir / 'test_model.json'

			paths = {'dir': model_dir, 'meta': meta_path}

			# Save
			fitted_model.save(paths, extra_meta={'version': '1.0'})

			# Load
			loaded_model = SklearnMultinomialClassifier.load(paths)

			assert loaded_model is not None
			assert hasattr(loaded_model.model, 'coef_')
			assert hasattr(loaded_model.scaler, 'mean_')

	def test_load_predictions_match_original(self, fitted_model):
		"""Test that loaded model makes same predictions."""
		with tempfile.TemporaryDirectory() as tmpdir:
			model_dir = Path(tmpdir) / 'models'
			model_dir.mkdir(parents=True)
			meta_path = model_dir / 'test_model.json'

			paths = {'dir': model_dir, 'meta': meta_path}

			# Get predictions from original
			np.random.seed(100)
			X_test = np.random.randn(30, 8).astype(np.float32)
			original_proba = fitted_model.predict_proba(X_test)
			original_class = fitted_model.predict_class(X_test)

			# Save and load
			fitted_model.save(paths, extra_meta={'version': '1.0'})
			loaded_model = SklearnMultinomialClassifier.load(paths)

			# Get predictions from loaded
			loaded_proba = loaded_model.predict_proba(X_test)
			loaded_class = loaded_model.predict_class(X_test)

			# Compare
			np.testing.assert_array_almost_equal(original_proba, loaded_proba, decimal=5)
			np.testing.assert_array_equal(original_class, loaded_class)

	def test_load_feature_scaling_preserved(self, fitted_model):
		"""Test that feature scaling is preserved after load."""
		with tempfile.TemporaryDirectory() as tmpdir:
			model_dir = Path(tmpdir) / 'models'
			model_dir.mkdir(parents=True)
			meta_path = model_dir / 'test_model.json'

			paths = {'dir': model_dir, 'meta': meta_path}

			# Save
			fitted_model.save(paths, extra_meta={'version': '1.0'})

			# Load
			loaded_model = SklearnMultinomialClassifier.load(paths)

			# Check scaling parameters
			np.testing.assert_array_almost_equal(fitted_model.scaler.mean_, loaded_model.scaler.mean_)
			np.testing.assert_array_almost_equal(fitted_model.scaler.scale_, loaded_model.scaler.scale_)

	def test_load_preserves_tag(self):
		"""Test that loaded model has correct tag."""
		model = SklearnMultinomialClassifier()
		np.random.seed(42)
		X = np.random.randn(50, 5).astype(np.float32)
		y = np.tile([0, 1, 2], 17)[:50]
		model.fit(X, y)

		with tempfile.TemporaryDirectory() as tmpdir:
			model_dir = Path(tmpdir) / 'models'
			model_dir.mkdir(parents=True)
			meta_path = model_dir / 'test_model.json'

			paths = {'dir': model_dir, 'meta': meta_path}
			model.save(paths, extra_meta={})

			loaded_model = SklearnMultinomialClassifier.load(paths)
			assert loaded_model.tag == 'multi_log_regression'


class TestSklearnMultinomialClassifierEdgeCases:
	"""Test edge cases and boundary conditions."""

	def test_minimal_data(self):
		"""Test fitting with minimal data."""
		model = SklearnMultinomialClassifier()
		X = np.random.randn(10, 2).astype(np.float32)
		y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])

		model.fit(X, y)
		predictions = model.predict(X)
		assert predictions.shape == (10,)

	def test_large_data(self):
		"""Test with larger dataset."""
		model = SklearnMultinomialClassifier()
		np.random.seed(42)
		X = np.random.randn(10000, 50).astype(np.float32)
		y = np.tile([0, 1, 2], 3334)[:10000]

		model.fit(X, y)
		assert hasattr(model.model, 'coef_')

		proba = model.predict_proba(X[:100])
		assert proba.shape == (100, 3)

	def test_single_feature(self):
		"""Test fitting with single feature."""
		model = SklearnMultinomialClassifier()
		X = np.random.randn(50, 1).astype(np.float32)
		y = np.tile([0, 1, 2], 17)[:50]

		model.fit(X, y)
		assert model.model.n_features_in_ == 1

	def test_all_same_dosage_raises_error(self):
		"""Test that all same dosage raises ValueError (sklearn requirement)."""
		model = SklearnMultinomialClassifier()
		X = np.random.randn(30, 5).astype(np.float32)
		y = np.zeros(30)  # All dosage 0

		# sklearn LogisticRegression requires 2+ classes
		with pytest.raises(ValueError, match='at least 2 classes'):
			model.fit(X, y)

	def test_zero_variance_data(self):
		"""Test with zero variance features."""
		model = SklearnMultinomialClassifier()
		X = np.ones((30, 5), dtype=np.float32)
		y = np.tile([0, 1, 2], 10)

		# Should handle gracefully
		model.fit(X, y)
		predictions = model.predict(X)
		assert predictions.shape == (30,)

	def test_extreme_values(self):
		"""Test with extreme input values."""
		model = SklearnMultinomialClassifier()
		np.random.seed(42)
		X = np.concatenate(
			[np.full((10, 5), -1e6, dtype=np.float32), np.full((10, 5), 1e6, dtype=np.float32), np.random.randn(30, 5).astype(np.float32)]
		)
		y = np.tile([0, 1, 2], 17)[:50]

		model.fit(X, y)
		predictions = model.predict(X)
		assert np.all(np.isfinite(predictions))

	def test_unbalanced_classes(self):
		"""Test with highly unbalanced dosage distribution."""
		model = SklearnMultinomialClassifier()
		X = np.random.randn(100, 5).astype(np.float32)
		y = np.concatenate([np.full(80, 0), np.full(15, 1), np.full(5, 2)])

		model.fit(X, y)
		assert hasattr(model.model, 'coef_')


class TestSklearnMultinomialClassifierIntegration:
	"""Integration tests for complete workflows."""

	def test_full_pipeline(self):
		"""Test complete fit-predict-save-load pipeline."""
		# Create and fit model
		model = SklearnMultinomialClassifier(random_seed=42)
		np.random.seed(42)
		X_train = np.random.randn(100, 10).astype(np.float32)
		y_train = np.tile([0, 1, 2], 34)[:100]

		model.fit(X_train, y_train)

		# Get predictions
		X_test = np.random.randn(20, 10).astype(np.float32)
		original_predictions = model.predict(X_test)

		# Save and load
		with tempfile.TemporaryDirectory() as tmpdir:
			model_dir = Path(tmpdir) / 'models'
			model_dir.mkdir(parents=True)
			meta_path = model_dir / 'test_model.json'

			paths = {'dir': model_dir, 'meta': meta_path}
			model.save(paths, extra_meta={'version': '1.0'})
			loaded_model = SklearnMultinomialClassifier.load(paths)

		# Verify loaded model
		loaded_predictions = loaded_model.predict(X_test)
		np.testing.assert_array_almost_equal(original_predictions, loaded_predictions, decimal=5)

	def test_multiple_save_load_cycles(self):
		"""Test repeated save/load cycles preserve predictions."""
		model = SklearnMultinomialClassifier(random_seed=42)
		np.random.seed(42)
		X_train = np.random.randn(100, 8).astype(np.float32)
		y_train = np.tile([0, 1, 2], 34)[:100]

		model.fit(X_train, y_train)

		X_test = np.random.randn(20, 8).astype(np.float32)
		original_predictions = model.predict(X_test)

		# Cycle through save/load
		current_model = model
		for cycle in range(3):
			with tempfile.TemporaryDirectory() as tmpdir:
				model_dir = Path(tmpdir) / 'models'
				model_dir.mkdir(parents=True)
				meta_path = model_dir / f'model_cycle_{cycle}.json'

				paths = {'dir': model_dir, 'meta': meta_path}
				current_model.save(paths, extra_meta={'cycle': cycle})
				current_model = SklearnMultinomialClassifier.load(paths)

		final_predictions = current_model.predict(X_test)
		np.testing.assert_array_almost_equal(original_predictions, final_predictions, decimal=5)

	def test_cross_validation_workflow(self):
		"""Test model can be used in cross-validation setup."""
		np.random.seed(42)
		X = np.random.randn(100, 8).astype(np.float32)
		y = np.tile([0, 1, 2], 34)[:100]

		# Simple 2-fold split
		split_idx = 50
		X_train1, X_test1 = X[:split_idx], X[split_idx:]
		y_train1, y_test1 = y[:split_idx], y[split_idx:]

		# Train fold 1
		model1 = SklearnMultinomialClassifier(random_seed=42)
		model1.fit(X_train1, y_train1)
		pred1 = model1.predict(X_test1)

		# Train fold 2 (with swapped data)
		model2 = SklearnMultinomialClassifier(random_seed=42)
		model2.fit(X_test1, y_test1)
		pred2 = model2.predict(X_train1)

		assert pred1.shape == (50,)
		assert pred2.shape == (50,)
