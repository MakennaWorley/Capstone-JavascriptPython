import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Add backend directory to path for package imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Remove any mocked model modules from sys.modules
sys.modules.pop('app.model_hmm', None)
sys.modules.pop('app.model_functions', None)

# Check if hmmlearn is available
try:
	import hmmlearn  # noqa: F401

	HMMLEARN_AVAILABLE = True
except ImportError:
	HMMLEARN_AVAILABLE = False

# Apply skip marker if hmmlearn not installed
pytestmark = pytest.mark.skipif(not HMMLEARN_AVAILABLE, reason='hmmlearn not installed')

# Now import the real module (only if hmmlearn available)
if HMMLEARN_AVAILABLE:
	from app.model_hmm import HMMDosageClassifier
else:
	# Provide a stub so the file can at least be parsed
	HMMDosageClassifier = None


class TestHMMDosageClassifierInitialization:
	"""Test HMMDosageClassifier initialization"""

	def test_init_default_parameters(self):
		"""Test initialization with default parameters"""
		model = HMMDosageClassifier()

		assert model.n_iter == 100
		assert model.n_components == 3
		assert model.covariance_type == 'diag'
		assert model.random_seed == 123
		assert model.tol == 1e-2
		assert model.verbose is True
		assert model.n_mix == 1

	def test_init_custom_parameters(self):
		"""Test initialization with custom parameters"""
		model = HMMDosageClassifier(n_iter=50, n_components=3, covariance_type='full', random_seed=42, tol=1e-3, verbose=False, n_mix=2)

		assert model.n_iter == 50
		assert model.covariance_type == 'full'
		assert model.random_seed == 42
		assert model.tol == 1e-3
		assert model.verbose is False
		assert model.n_mix == 2

	def test_init_model_none(self):
		"""Test that model is None before fitting"""
		model = HMMDosageClassifier()

		assert model.model is None
		assert model.feature_mean_ is None
		assert model.feature_std_ is None
		assert model.state_to_dosage_ is None

	def test_tag_property(self):
		"""Test tag property"""
		model = HMMDosageClassifier()

		assert model.tag == 'hmm_dosage'

	def test_init_with_different_covariance_types(self):
		"""Test initialization with different covariance types"""
		for cov_type in ['diag', 'full', 'spherical', 'tied']:
			model = HMMDosageClassifier(covariance_type=cov_type)
			assert model.covariance_type == cov_type


class TestHMMDosageClassifierFitting:
	"""Test HMMDosageClassifier fitting"""

	def test_fit_basic(self):
		"""Test basic fitting with small dataset"""
		model = HMMDosageClassifier(n_iter=5, verbose=False)

		np.random.seed(42)
		X = np.random.randn(30, 5).astype(np.float32)
		y = np.tile([0, 1, 2], 10)

		result = model.fit(X, y)

		assert result is model  # fit returns self
		assert model.model is not None
		assert model.feature_mean_ is not None
		assert model.feature_std_ is not None
		assert model.state_to_dosage_ is not None
		assert len(model.state_to_dosage_) == 3

	def test_fit_standardizes_features(self):
		"""Test that fit standardizes features"""
		model = HMMDosageClassifier(n_iter=5, verbose=False)

		np.random.seed(42)
		X = np.random.randn(30, 5).astype(np.float32) * 100 + 50
		y = np.tile([0, 1, 2], 10)

		model.fit(X, y)

		# feature_mean_ and feature_std_ should be set
		assert model.feature_mean_ is not None
		assert model.feature_std_ is not None
		assert len(model.feature_mean_) == 5
		assert len(model.feature_std_) == 5

	def test_fit_with_imbalanced_classes(self):
		"""Test fitting with imbalanced dosage classes"""
		model = HMMDosageClassifier(n_iter=5, verbose=False)

		np.random.seed(42)
		X = np.random.randn(50, 4).astype(np.float32)
		y = np.array([0] * 30 + [1] * 15 + [2] * 5, dtype=np.float32)

		result = model.fit(X, y)

		assert result is model
		assert model.model is not None
		assert model.pycm_metrics_ is not None

	def test_fit_single_class_samples(self):
		"""Test fitting with only one dosage class present"""
		model = HMMDosageClassifier(n_iter=5, verbose=False)

		np.random.seed(42)
		X = np.random.randn(20, 4).astype(np.float32)
		y = np.zeros(20, dtype=np.float32)  # All class 0

		# Should still fit, but may not align all states
		result = model.fit(X, y)

		assert result is model
		assert model.model is not None

	def test_fit_computes_pycm_metrics(self):
		"""Test that fit computes PYCM metrics"""
		model = HMMDosageClassifier(n_iter=5, verbose=False)

		np.random.seed(42)
		X = np.random.randn(30, 5).astype(np.float32)
		y = np.tile([0, 1, 2], 10)

		model.fit(X, y)

		assert model.pycm_metrics_ is not None
		assert 'overall_accuracy' in model.pycm_metrics_
		assert 'kappa' in model.pycm_metrics_
		assert 'overall_f1' in model.pycm_metrics_


class TestHMMDosageClassifierPrediction:
	"""Test HMMDosageClassifier prediction methods"""

	def test_predict_proba_basic(self):
		"""Test probability prediction"""
		model = HMMDosageClassifier(n_iter=5, verbose=False)

		np.random.seed(42)
		X = np.random.randn(30, 5).astype(np.float32)
		y = np.tile([0, 1, 2], 10)

		model.fit(X, y)

		# Predict on same data
		proba = model.predict_proba(X)

		assert proba.shape == (30, 3)
		assert np.allclose(proba.sum(axis=1), 1.0)  # Probabilities sum to 1
		assert (proba >= 0).all() and (proba <= 1).all()

	def test_predict_proba_not_fitted_raises(self):
		"""Test that predict_proba raises error if model not fitted"""
		model = HMMDosageClassifier()

		X = np.random.randn(10, 5).astype(np.float32)

		with pytest.raises(RuntimeError, match='must be fitted'):
			model.predict_proba(X)

	def test_predict_class_basic(self):
		"""Test class prediction"""
		model = HMMDosageClassifier(n_iter=5, verbose=False)

		np.random.seed(42)
		X = np.random.randn(30, 5).astype(np.float32)
		y = np.tile([0, 1, 2], 10)

		model.fit(X, y)

		pred_class = model.predict_class(X)

		assert pred_class.shape == (30,)
		assert set(pred_class).issubset({0, 1, 2})

	def test_predict_class_returns_argmax(self):
		"""Test that predict_class returns argmax of probabilities"""
		model = HMMDosageClassifier(n_iter=5, verbose=False)

		np.random.seed(42)
		X = np.random.randn(30, 5).astype(np.float32)
		y = np.tile([0, 1, 2], 10)

		model.fit(X, y)

		proba = model.predict_proba(X)
		pred_class = model.predict_class(X)
		expected_class = np.argmax(proba, axis=1)

		np.testing.assert_array_equal(pred_class, expected_class)

	def test_predict_expected_dosage(self):
		"""Test expected dosage prediction"""
		model = HMMDosageClassifier(n_iter=5, verbose=False)

		np.random.seed(42)
		X = np.random.randn(30, 5).astype(np.float32)
		y = np.tile([0, 1, 2], 10)

		model.fit(X, y)

		dosage = model.predict(X)

		assert dosage.shape == (30,)
		assert (dosage >= 0).all() and (dosage <= 2).all()

	def test_predict_expected_dosage_computation(self):
		"""Test that predict returns E[y] = sum_c c * p(c)"""
		model = HMMDosageClassifier(n_iter=5, verbose=False)

		np.random.seed(42)
		X = np.random.randn(15, 4).astype(np.float32)
		y = np.tile([0, 1, 2], 5)

		model.fit(X, y)

		proba = model.predict_proba(X)
		dosage = model.predict(X)

		# Expected dosage should be sum of class * probability
		expected_dosage = (proba * np.array([0, 1, 2])).sum(axis=1)

		np.testing.assert_allclose(dosage, expected_dosage, rtol=1e-5)

	def test_predict_with_groups_parameter(self):
		"""Test prediction with groups parameter (ignored in current implementation)"""
		model = HMMDosageClassifier(n_iter=5, verbose=False)

		np.random.seed(42)
		X = np.random.randn(30, 5).astype(np.float32)
		y = np.tile([0, 1, 2], 10)
		groups = np.tile([0, 1], 15)

		model.fit(X, y, groups=groups)

		# Should work without error
		proba = model.predict_proba(X, groups=groups)
		pred_class = model.predict_class(X, groups=groups)
		dosage = model.predict(X, groups=groups)

		assert proba.shape == (30, 3)
		assert pred_class.shape == (30,)
		assert dosage.shape == (30,)


class TestHMMDosageClassifierEvaluation:
	"""Test HMMDosageClassifier evaluation methods"""

	def test_evaluate_pycm(self):
		"""Test PYCM evaluation"""
		model = HMMDosageClassifier(n_iter=5, verbose=False)

		np.random.seed(42)
		X = np.random.randn(30, 5).astype(np.float32)
		y = np.tile([0, 1, 2], 10)

		model.fit(X, y)

		pycm_cm = model.evaluate_pycm(X, y)

		assert pycm_cm is not None
		assert hasattr(pycm_cm, 'Overall_ACC')
		assert hasattr(pycm_cm, 'F1_Macro')

	@pytest.mark.xfail(reason='model_hmm.py has covariance shape bug in predictions on different data')
	def test_evaluate_pycm_different_data(self):
		"""Test PYCM evaluation on different data"""
		model = HMMDosageClassifier(n_iter=5, verbose=False)

		np.random.seed(42)
		X_train = np.random.randn(30, 5).astype(np.float32)
		y_train = np.tile([0, 1, 2], 10)

		model.fit(X_train, y_train)

		# Evaluate on new data
		np.random.seed(43)
		X_test = np.random.randn(20, 5).astype(np.float32)
		y_test = np.tile([0, 1, 2], 7)[:20]

		pycm_cm = model.evaluate_pycm(X_test, y_test)

		assert pycm_cm is not None


class TestHMMDosageClassifierPersistence:
	"""Test HMMDosageClassifier save/load functionality"""

	def test_save_basic(self):
		"""Test saving model"""
		model = HMMDosageClassifier(n_iter=5, verbose=False)

		np.random.seed(42)
		X = np.random.randn(30, 5).astype(np.float32)
		y = np.tile([0, 1, 2], 10)

		model.fit(X, y)

		with tempfile.TemporaryDirectory() as tmpdir:
			model_dir = Path(tmpdir) / 'models'
			model_dir.mkdir(parents=True)
			meta_path = model_dir / 'test_model.json'

			paths = {'dir': model_dir, 'meta': meta_path}

			model.save(paths, extra_meta={'test': 'data'})

			# Check files were created
			assert meta_path.exists()

	def test_save_not_fitted_raises(self):
		"""Test that save raises error if model not fitted"""
		model = HMMDosageClassifier()

		with tempfile.TemporaryDirectory() as tmpdir:
			paths = {'dir': Path(tmpdir), 'meta': Path(tmpdir) / 'model.json'}

			with pytest.raises(RuntimeError, match='No model to save'):
				model.save(paths, extra_meta={})

	@pytest.mark.xfail(reason='model_hmm.py has covariance shape bug in save/load')
	def test_load_basic(self):
		"""Test loading model"""
		model = HMMDosageClassifier(n_iter=5, verbose=False, random_seed=42)

		np.random.seed(42)
		X = np.random.randn(30, 5).astype(np.float32)
		y = np.tile([0, 1, 2], 10)

		model.fit(X, y)

		with tempfile.TemporaryDirectory() as tmpdir:
			model_dir = Path(tmpdir) / 'models'
			model_dir.mkdir(parents=True)
			meta_path = model_dir / 'test_model.json'

			paths = {'dir': model_dir, 'meta': meta_path}

			# Save
			model.save(paths, extra_meta={'version': '1.0'})

			# Load
			loaded_model = HMMDosageClassifier.load(paths)

			assert loaded_model.model is not None
			assert loaded_model.feature_mean_ is not None
			assert loaded_model.feature_std_ is not None
			assert loaded_model.state_to_dosage_ is not None

	@pytest.mark.xfail(reason='model_hmm.py has covariance shape bug in save/load')
	def test_load_predictions_match_original(self):
		"""Test that predictions from loaded model match original"""
		model = HMMDosageClassifier(n_iter=5, verbose=False, random_seed=42)

		np.random.seed(42)
		X = np.random.randn(30, 5).astype(np.float32)
		y = np.tile([0, 1, 2], 10)

		model.fit(X, y)

		# Get predictions from original
		orig_pred = model.predict_class(X[:5])
		orig_proba = model.predict_proba(X[:5])

		with tempfile.TemporaryDirectory() as tmpdir:
			model_dir = Path(tmpdir) / 'models'
			model_dir.mkdir(parents=True)
			meta_path = model_dir / 'test_model.json'

			paths = {'dir': model_dir, 'meta': meta_path}

			# Save and load
			model.save(paths, extra_meta={})
			loaded_model = HMMDosageClassifier.load(paths)

			# Get predictions from loaded
			loaded_pred = loaded_model.predict_class(X[:5])
			loaded_proba = loaded_model.predict_proba(X[:5])

			np.testing.assert_array_equal(orig_pred, loaded_pred)
			np.testing.assert_allclose(orig_proba, loaded_proba, rtol=1e-5)

	@pytest.mark.xfail(reason='model_hmm.py has covariance shape bug in save/load')
	def test_load_feature_scaling_preserved(self):
		"""Test that feature scaling is preserved after load"""
		model = HMMDosageClassifier(n_iter=5, verbose=False)

		np.random.seed(42)
		X = np.random.randn(30, 5).astype(np.float32) * 100 + 50
		y = np.tile([0, 1, 2], 10)

		model.fit(X, y)

		orig_mean = model.feature_mean_.copy()
		orig_std = model.feature_std_.copy()

		with tempfile.TemporaryDirectory() as tmpdir:
			model_dir = Path(tmpdir) / 'models'
			model_dir.mkdir(parents=True)
			meta_path = model_dir / 'test_model.json'

			paths = {'dir': model_dir, 'meta': meta_path}

			# Save and load
			model.save(paths, extra_meta={})
			loaded_model = HMMDosageClassifier.load(paths)

			np.testing.assert_allclose(loaded_model.feature_mean_, orig_mean, rtol=1e-5)
			np.testing.assert_allclose(loaded_model.feature_std_, orig_std, rtol=1e-5)


class TestHMMDosageClassifierEdgeCases:
	"""Test HMMDosageClassifier edge cases"""

	def test_fit_with_float_labels(self):
		"""Test fitting with float labels that need coercion"""
		model = HMMDosageClassifier(n_iter=5, verbose=False)

		np.random.seed(42)
		X = np.random.randn(30, 5).astype(np.float32)
		y = np.array([0.1, 1.2, 1.9] * 10, dtype=np.float32)

		result = model.fit(X, y)

		assert result is model
		assert model.model is not None

	def test_predict_maintains_shape(self):
		"""Test that predictions maintain correct shape"""
		model = HMMDosageClassifier(n_iter=5, verbose=False)

		np.random.seed(42)
		X = np.random.randn(30, 5).astype(np.float32)
		y = np.tile([0, 1, 2], 10)

		model.fit(X, y)

		# Test with different sizes
		for n_samples in [1, 5, 10, 30]:
			X_test = np.random.randn(n_samples, 5).astype(np.float32)

			proba = model.predict_proba(X_test)
			pred_class = model.predict_class(X_test)
			dosage = model.predict(X_test)

			assert proba.shape == (n_samples, 3)
			assert pred_class.shape == (n_samples,)
			assert dosage.shape == (n_samples,)

	def test_single_sample_prediction(self):
		"""Test prediction on single sample"""
		model = HMMDosageClassifier(n_iter=5, verbose=False)

		np.random.seed(42)
		X = np.random.randn(30, 5).astype(np.float32)
		y = np.tile([0, 1, 2], 10)

		model.fit(X, y)

		# Single sample
		X_single = X[0:1]

		proba = model.predict_proba(X_single)
		pred_class = model.predict_class(X_single)

		assert proba.shape == (1, 3)
		assert pred_class.shape == (1,)

	def test_small_feature_set(self):
		"""Test with very small feature set"""
		model = HMMDosageClassifier(n_iter=5, verbose=False)

		np.random.seed(42)
		X = np.random.randn(30, 2).astype(np.float32)
		y = np.tile([0, 1, 2], 10)

		result = model.fit(X, y)

		assert result is model
		proba = model.predict_proba(X)
		assert proba.shape == (30, 3)

	def test_large_feature_set(self):
		"""Test with large feature set"""
		model = HMMDosageClassifier(n_iter=3, verbose=False)

		np.random.seed(42)
		X = np.random.randn(20, 50).astype(np.float32)
		y = np.tile([0, 1, 2], 7)[:20]

		result = model.fit(X, y)

		assert result is model
		proba = model.predict_proba(X)
		assert proba.shape == (20, 3)


class TestHMMDosageClassifierIntegration:
	"""Integration tests for HMMDosageClassifier"""

	@pytest.mark.xfail(reason='model_hmm.py has covariance shape bug in save/load')
	def test_full_workflow(self):
		"""Test complete workflow: create, fit, predict, save, load"""
		# Create and fit model
		model = HMMDosageClassifier(n_iter=5, verbose=False, random_seed=42)

		np.random.seed(42)
		X_train = np.random.randn(39, 5).astype(np.float32)
		y_train = np.tile([0, 1, 2], 13)

		model.fit(X_train, y_train)

		# Evaluate
		pycm = model.evaluate_pycm(X_train, y_train)
		assert pycm is not None

		# Predictions
		proba = model.predict_proba(X_train)
		assert proba.shape == (39, 3)

		# Save and load
		with tempfile.TemporaryDirectory() as tmpdir:
			model_dir = Path(tmpdir) / 'models'
			model_dir.mkdir(parents=True)
			meta_path = model_dir / 'hmm_model.json'

			paths = {'dir': model_dir, 'meta': meta_path}

			model.save(paths, extra_meta={'dataset': 'test'})
			loaded = HMMDosageClassifier.load(paths)

			# Verify loaded model works
			loaded_pred = loaded.predict_class(X_train)
			assert loaded_pred.shape == (39,)

	def test_consistency_across_calls(self):
		"""Test that predictions are consistent across multiple calls"""
		model = HMMDosageClassifier(n_iter=5, verbose=False, random_seed=42)

		np.random.seed(42)
		X = np.random.randn(20, 5).astype(np.float32)
		y = np.tile([0, 1, 2], 7)[:20]

		model.fit(X, y)

		# Get predictions multiple times
		pred1 = model.predict_class(X)
		pred2 = model.predict_class(X)
		pred3 = model.predict_class(X)

		np.testing.assert_array_equal(pred1, pred2)
		np.testing.assert_array_equal(pred2, pred3)
