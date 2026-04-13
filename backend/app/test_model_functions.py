import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Add backend directory to path for package imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Remove any mocked model_functions from sys.modules
sys.modules.pop('app.model_functions', None)

# Now import the real module
from app.model_functions import (
	NoRefitProxy,
	coerce_dosage_classes,
	ensure_dir,
	flatten_examples,
	load_meta,
	model_paths,
	save_common_meta,
	standardize_apply,
	standardize_fit,
)


class TestNoRefitProxy:
	"""Test NoRefitProxy wrapper class"""

	def test_proxy_wraps_model(self):
		"""Test that proxy wraps a fitted model"""

		class DummyModel:
			def fit(self, X, y):
				return self

			def predict(self, X):
				return np.array([1, 2, 3])

		model = DummyModel()
		proxy = NoRefitProxy(model)
		assert proxy._m is model

	def test_proxy_fit_is_noop(self):
		"""Test that fit() returns self without refitting"""

		class DummyModel:
			fit_count = 0

			def fit(self, X, y):
				self.fit_count += 1
				return self

			def predict(self, X):
				return np.array([1, 2, 3])

		model = DummyModel()
		model.fit(np.array([[1, 2]]), np.array([0]))  # Fit once
		initial_count = model.fit_count

		proxy = NoRefitProxy(model)
		result = proxy.fit(np.array([[3, 4]]), np.array([1]))

		# Fit count should not increase; fit() should be a no-op
		assert result is proxy
		assert model.fit_count == initial_count

	def test_proxy_predict_delegates(self):
		"""Test that predict() delegates to wrapped model"""

		class DummyModel:
			def fit(self, X, y):
				return self

			def predict(self, X):
				return np.array([1, 2, 3])

		model = DummyModel()
		proxy = NoRefitProxy(model)
		X = np.array([[1, 2], [3, 4]])

		result = proxy.predict(X)
		expected = model.predict(X)

		assert np.array_equal(result, expected)


class TestEnsureDir:
	"""Test ensure_dir directory creation"""

	def test_ensure_dir_creates_single_dir(self):
		"""Test creating a single directory"""
		with tempfile.TemporaryDirectory() as tmpdir:
			test_path = Path(tmpdir) / 'test_dir'
			assert not test_path.exists()

			ensure_dir(test_path)

			assert test_path.exists()
			assert test_path.is_dir()

	def test_ensure_dir_creates_nested_dirs(self):
		"""Test creating nested directory structure"""
		with tempfile.TemporaryDirectory() as tmpdir:
			nested_path = Path(tmpdir) / 'level1' / 'level2' / 'level3'
			assert not nested_path.exists()

			ensure_dir(nested_path)

			assert nested_path.exists()
			assert nested_path.is_dir()

	def test_ensure_dir_idempotent(self):
		"""Test that ensure_dir is idempotent"""
		with tempfile.TemporaryDirectory() as tmpdir:
			test_path = Path(tmpdir) / 'test_dir'

			ensure_dir(test_path)
			ensure_dir(test_path)  # Call twice
			ensure_dir(test_path)  # Call three times

			assert test_path.exists()

	def test_ensure_dir_with_string_path(self):
		"""Test ensure_dir with string path"""
		with tempfile.TemporaryDirectory() as tmpdir:
			test_path = str(Path(tmpdir) / 'test_dir')
			assert not Path(test_path).exists()

			ensure_dir(test_path)

			assert Path(test_path).exists()


class TestFlattenExamples:
	"""Test flatten_examples function"""

	def test_flatten_examples_basic(self):
		"""Test basic flattening of 3D to 2D"""
		X = np.array([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]], dtype=np.float32)
		y = np.array([[0, 0, 1], [1, 2, 0]], dtype=np.float32)

		X_flat, y_flat = flatten_examples(X, y)

		assert X_flat.shape == (6, 2)
		assert y_flat.shape == (6,)
		assert np.array_equal(X_flat[0], [1, 2])
		assert np.array_equal(X_flat[3], [7, 8])
		assert y_flat[0] == 0
		assert y_flat[3] == 1

	def test_flatten_examples_single_sample(self):
		"""Test flattening with single sample"""
		X = np.array([[[1, 2], [3, 4]]], dtype=np.float32)
		y = np.array([[0, 1]], dtype=np.float32)

		X_flat, y_flat = flatten_examples(X, y)

		assert X_flat.shape == (2, 2)
		assert y_flat.shape == (2,)

	def test_flatten_examples_wrong_x_rank_raises(self):
		"""Test that wrong X rank raises error"""
		X = np.array([[1, 2], [3, 4]], dtype=np.float32)  # 2D instead of 3D
		y = np.array([[0, 1], [1, 0]], dtype=np.float32)

		with pytest.raises(ValueError, match='Expected X rank-3'):
			flatten_examples(X, y)

	def test_flatten_examples_wrong_y_rank_raises(self):
		"""Test that wrong y rank raises error"""
		X = np.array([[[1, 2], [3, 4]]], dtype=np.float32)  # 3D
		y = np.array([0, 1], dtype=np.float32)  # 1D instead of 2D

		with pytest.raises(ValueError, match='Expected y rank-2'):
			flatten_examples(X, y)

	def test_flatten_examples_shape_mismatch_raises(self):
		"""Test that shape mismatch raises error"""
		X = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)  # 2x2x2
		y = np.array([[0, 1]], dtype=np.float32)  # 1x2 instead of 2x2

		with pytest.raises(ValueError, match='X/y mismatch'):
			flatten_examples(X, y)

	def test_flatten_examples_returns_float32(self):
		"""Test that output is float32"""
		X = np.array([[[1, 2], [3, 4]]], dtype=np.int32)
		y = np.array([[0, 1]], dtype=np.int32)

		X_flat, y_flat = flatten_examples(X, y)

		assert X_flat.dtype == np.float32
		assert y_flat.dtype == np.float32


class TestStandardizeFit:
	"""Test standardize_fit function"""

	def test_standardize_fit_basic(self):
		"""Test basic standardization fitting"""
		X = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)

		X_std, mu, sd = standardize_fit(X)

		assert X_std.shape == X.shape
		assert mu.shape == (2,)
		assert sd.shape == (2,)
		assert np.allclose(mu, [3, 4])  # Mean along axis 0
		# Std for column 0: sqrt(((-2)^2 + 0^2 + 2^2) / 3) = sqrt(8/3) ≈ 1.633
		expected_sd = np.sqrt(8 / 3)
		assert np.allclose(sd, [expected_sd, expected_sd], atol=0.01)

	def test_standardize_fit_with_zero_std(self):
		"""Test standardization when std is zero"""
		X = np.array([[1, 2], [1, 2], [1, 2]], dtype=np.float32)

		X_std, mu, sd = standardize_fit(X)

		# Where std is 0, it should be replaced with 1
		assert np.all(sd > 0)
		assert sd[0] == 1.0

	def test_standardize_fit_centered(self):
		"""Test that standardized values are centered around 0"""
		X = np.array([[1, 10], [2, 20], [3, 30]], dtype=np.float32)
		X_std, mu, sd = standardize_fit(X)

		# Mean of standardized should be close to 0
		assert np.allclose(X_std.mean(axis=0), [0, 0], atol=1e-6)

	def test_standardize_fit_unit_variance(self):
		"""Test that standardized has unit variance (approximately)"""
		X = np.array([[1, 10], [2, 20], [3, 30], [4, 40]], dtype=np.float32)
		X_std, mu, sd = standardize_fit(X)

		# Variance of standardized should be close to 1
		var = X_std.var(axis=0)
		assert np.allclose(var, [1, 1], atol=0.1)


class TestStandardizeApply:
	"""Test standardize_apply function"""

	def test_standardize_apply_basic(self):
		"""Test applying standardization"""
		X = np.array([[1, 2], [3, 4]], dtype=np.float32)
		mu = np.array([2, 3], dtype=np.float32)
		sd = np.array([1, 1], dtype=np.float32)

		X_std = standardize_apply(X, mu, sd)

		expected = np.array([[-1, -1], [1, 1]], dtype=np.float32)
		assert np.allclose(X_std, expected)

	def test_standardize_apply_matches_fit(self):
		"""Test that apply matches fit results"""
		X_train = np.array([[1, 10], [2, 20], [3, 30]], dtype=np.float32)
		X_test = np.array([[2, 15], [3, 25]], dtype=np.float32)

		X_train_std, mu, sd = standardize_fit(X_train)
		X_test_std = standardize_apply(X_test, mu, sd)

		# Verify train standardization is consistent
		X_train_std_verify = standardize_apply(X_train, mu, sd)
		assert np.allclose(X_train_std, X_train_std_verify)

	def test_standardize_apply_known_values(self):
		"""Test standardize_apply with explicit mu/sd produces correct values"""
		X = np.array([[4.0, 100.0], [6.0, 200.0]], dtype=np.float32)
		mu = np.array([2.0, 50.0], dtype=np.float32)
		sd = np.array([2.0, 50.0], dtype=np.float32)

		X_std = standardize_apply(X, mu, sd)

		# (4-2)/2=1, (100-50)/50=1, (6-2)/2=2, (200-50)/50=3
		expected = np.array([[1.0, 1.0], [2.0, 3.0]], dtype=np.float32)
		assert np.allclose(X_std, expected)


class TestCoerceDosageClasses:
	"""Test coerce_dosage_classes function"""

	def test_coerce_exact_integers(self):
		"""Test coercion of exact integer values"""
		y = np.array([0.0, 1.0, 2.0], dtype=np.float32)

		y_coerced = coerce_dosage_classes(y)

		assert np.array_equal(y_coerced, [0, 1, 2])
		assert y_coerced.dtype == np.int64

	def test_coerce_rounds_to_nearest(self):
		"""Test that values are rounded to nearest integer"""
		y = np.array([0.1, 0.9, 1.4, 1.6, 2.1], dtype=np.float32)

		y_coerced = coerce_dosage_classes(y)

		assert np.array_equal(y_coerced, [0, 1, 1, 2, 2])

	def test_coerce_clips_out_of_range(self):
		"""Test that out-of-range values are clipped"""
		y = np.array([-1.0, -0.5, 0.5, 2.5, 3.0, 5.0], dtype=np.float32)

		y_coerced = coerce_dosage_classes(y)

		# All values should be in {0, 1, 2}
		assert np.all(y_coerced >= 0)
		assert np.all(y_coerced <= 2)
		assert np.array_equal(y_coerced, [0, 0, 0, 2, 2, 2])

	def test_coerce_negative_values(self):
		"""Test coercion of negative values"""
		y = np.array([-5.0, -1.0, 0.0], dtype=np.float32)

		y_coerced = coerce_dosage_classes(y)

		assert np.all(y_coerced >= 0)

	def test_coerce_bankers_rounding_at_half(self):
		"""Test that np.rint uses banker's rounding (round-half-to-even)"""
		# np.rint(0.5) -> 0 (rounds to even), np.rint(1.5) -> 2 (rounds to even)
		# np.rint(2.5) -> 2 (rounds to even, then clipped to 2)
		y = np.array([0.5, 1.5, 2.5], dtype=np.float32)
		y_coerced = coerce_dosage_classes(y)
		assert np.array_equal(y_coerced, [0, 2, 2])


class TestModelPaths:
	"""Test model_paths function"""

	def test_model_paths_basic(self):
		"""Test basic path generation"""
		with tempfile.TemporaryDirectory() as tmpdir:
			models_dir = Path(tmpdir)
			base_name = 'test.training'
			model_tag = 'bayes_softmax3'

			paths = model_paths(models_dir, base_name, model_tag)

			assert 'dir' in paths
			assert 'idata' in paths
			assert 'meta' in paths
			assert 'graph_test' in paths
			assert 'graph_cm' in paths

	def test_model_paths_correct_names(self):
		"""Test that paths have correct filenames"""
		with tempfile.TemporaryDirectory() as tmpdir:
			models_dir = Path(tmpdir)
			base_name = 'test.training'
			model_tag = 'dnn_dosage'

			paths = model_paths(models_dir, base_name, model_tag)

			assert str(paths['idata']).endswith('.idata.nc')
			assert str(paths['meta']).endswith('.meta.json')
			assert str(paths['graph_test']).endswith('.test_plot.png')
			assert str(paths['graph_cm']).endswith('.cm_plot.png')

	def test_model_paths_with_string_path(self):
		"""Test model_paths with string input"""
		with tempfile.TemporaryDirectory() as tmpdir:
			models_dir = str(Path(tmpdir))
			base_name = 'dataset.split'
			model_tag = 'model_type'

			paths = model_paths(models_dir, base_name, model_tag)

			assert isinstance(paths['dir'], Path)
			assert paths['dir'] == Path(tmpdir)

	def test_model_paths_embeds_base_name_and_tag(self):
		"""Test that base_name and model_tag are embedded in all non-dir paths"""
		with tempfile.TemporaryDirectory() as tmpdir:
			base_name = 'run1.training'
			model_tag = 'bayes_softmax3'

			paths = model_paths(tmpdir, base_name, model_tag)

			for key in ('idata', 'meta', 'graph_test', 'graph_cm'):
				name = paths[key].name
				assert base_name in name, f'{key} path missing base_name'
				assert model_tag in name, f'{key} path missing model_tag'


class TestSaveAndLoadMeta:
	"""Test save_common_meta and load_meta functions"""

	def test_save_and_load_meta_basic(self):
		"""Test saving and loading metadata"""
		with tempfile.TemporaryDirectory() as tmpdir:
			tmpdir_path = Path(tmpdir)
			paths = {'dir': tmpdir_path, 'meta': tmpdir_path / 'meta.json'}

			payload = {'type': 'TestModel', 'tag': 'test_tag', 'extra': {'key': 'value', 'number': 42}}

			save_common_meta(paths, payload)
			loaded = load_meta(paths)

			assert loaded['type'] == 'TestModel'
			assert loaded['tag'] == 'test_tag'
			assert loaded['extra']['key'] == 'value'
			assert loaded['extra']['number'] == 42

	def test_save_meta_creates_file(self):
		"""Test that save_common_meta creates the file"""
		with tempfile.TemporaryDirectory() as tmpdir:
			tmpdir_path = Path(tmpdir)
			paths = {'dir': tmpdir_path, 'meta': tmpdir_path / 'meta.json'}

			payload = {'test': 'data'}

			assert not paths['meta'].exists()
			save_common_meta(paths, payload)
			assert paths['meta'].exists()

	def test_save_meta_with_nested_data(self):
		"""Test saving complex nested data structures"""
		with tempfile.TemporaryDirectory() as tmpdir:
			tmpdir_path = Path(tmpdir)
			paths = {'dir': tmpdir_path, 'meta': tmpdir_path / 'meta.json'}

			payload = {
				'params': {'layers': [256, 128, 64], 'dropout': 0.3, 'learning_rate': 0.001},
				'metrics': {'accuracy': [0.85, 0.90, 0.92], 'loss': [0.5, 0.3, 0.2]},
			}

			save_common_meta(paths, payload)
			loaded = load_meta(paths)

			assert loaded['params']['layers'] == [256, 128, 64]
			assert loaded['params']['dropout'] == 0.3
			assert loaded['metrics']['accuracy'] == [0.85, 0.90, 0.92]

	def test_save_meta_with_numpy_types(self):
		"""Test saving numpy types as JSON"""
		with tempfile.TemporaryDirectory() as tmpdir:
			tmpdir_path = Path(tmpdir)
			paths = {'dir': tmpdir_path, 'meta': tmpdir_path / 'meta.json'}

			payload = {
				'data': {
					'array': [1.0, 2.0, 3.0],  # Use native Python lists
					'mean': 2.0,
					'std': 0.816496580927726,
				}
			}

			save_common_meta(paths, payload)
			loaded = load_meta(paths)

			assert isinstance(loaded['data']['array'], list)
			assert len(loaded['data']['array']) == 3

	def test_load_meta_missing_file_raises(self):
		"""Test that load_meta raises error for missing file"""
		with tempfile.TemporaryDirectory() as tmpdir:
			tmpdir_path = Path(tmpdir)
			paths = {'meta': tmpdir_path / 'nonexistent.json'}

			with pytest.raises(FileNotFoundError):
				load_meta(paths)

	def test_save_meta_creates_dir_if_needed(self):
		"""Test that save_common_meta creates directory if needed"""
		with tempfile.TemporaryDirectory() as tmpdir:
			tmpdir_path = Path(tmpdir) / 'subdir'
			paths = {'dir': tmpdir_path, 'meta': tmpdir_path / 'meta.json'}

			assert not tmpdir_path.exists()

			save_common_meta(paths, {'test': 'data'})

			assert tmpdir_path.exists()
			assert paths['meta'].exists()

	def test_meta_json_format(self):
		"""Test that saved JSON is properly formatted"""
		with tempfile.TemporaryDirectory() as tmpdir:
			tmpdir_path = Path(tmpdir)
			paths = {'dir': tmpdir_path, 'meta': tmpdir_path / 'meta.json'}

			payload = {'type': 'Model', 'version': 1}
			save_common_meta(paths, payload)

			# Read raw JSON and verify format
			json_text = paths['meta'].read_text(encoding='utf-8')
			parsed = json.loads(json_text)

			assert parsed == payload
			# Verify it's indented (human readable)
			assert '\n' in json_text

	def test_save_meta_keys_sorted(self):
		"""Test that keys are written in sorted order (sort_keys=True)"""
		with tempfile.TemporaryDirectory() as tmpdir:
			tmpdir_path = Path(tmpdir)
			paths = {'dir': tmpdir_path, 'meta': tmpdir_path / 'meta.json'}

			payload = {'zebra': 1, 'alpha': 2, 'middle': 3}
			save_common_meta(paths, payload)

			json_text = paths['meta'].read_text(encoding='utf-8')
			# Keys must appear in alphabetical order in the raw text
			alpha_pos = json_text.index('alpha')
			middle_pos = json_text.index('middle')
			zebra_pos = json_text.index('zebra')
			assert alpha_pos < middle_pos < zebra_pos
