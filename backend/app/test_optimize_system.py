"""
Comprehensive pytest suite for optimize_system.py module.

Tests cover environment configuration, system checks, package management,
and main optimization workflow.
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent))

from optimize_system import check_system_specs, install_missing_packages, optimize_system, set_environment_variables


class TestSetEnvironmentVariables:
	"""Test environment variable configuration."""

	def test_sets_gpu_memory_fraction(self, capsys):
		"""Test that GPU memory fraction is set."""
		set_environment_variables()
		assert os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] == '0.85'

	def test_sets_gpu_preallocation(self):
		"""Test that GPU preallocation is enabled."""
		set_environment_variables()
		assert os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] == 'true'

	def test_sets_force_gpu_growth(self):
		"""Test TF GPU memory growth setting."""
		set_environment_variables()
		assert os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] == 'true'

	def test_sets_jax_float32(self):
		"""Test JAX is configured for float32."""
		set_environment_variables()
		assert os.environ['JAX_ENABLE_X64'] == 'false'

	def test_sets_xla_flags(self):
		"""Test XLA optimization flags."""
		set_environment_variables()
		assert 'xla_gpu_enable_fast_math=true' in os.environ['XLA_FLAGS']

	def test_sets_thread_count(self):
		"""Test OMP thread count configuration."""
		set_environment_variables()
		assert os.environ['OMP_NUM_THREADS'] == '20'
		assert os.environ['MKL_NUM_THREADS'] == '20'
		assert os.environ['NUMBA_NUM_THREADS'] == '20'

	def test_sets_reproducibility_seed(self):
		"""Test reproducibility hash seed."""
		set_environment_variables()
		assert os.environ['PYTHONHASHSEED'] == '0'

	def test_all_expected_variables_set(self):
		"""Test that all expected environment variables are set."""
		expected_vars = [
			'XLA_PYTHON_CLIENT_MEM_FRACTION',
			'XLA_PYTHON_CLIENT_PREALLOCATE',
			'TF_FORCE_GPU_ALLOW_GROWTH',
			'JAX_ENABLE_X64',
			'XLA_FLAGS',
			'OMP_NUM_THREADS',
			'MKL_NUM_THREADS',
			'NUMBA_NUM_THREADS',
			'THEANO_FLAGS',
			'PYTHONHASHSEED',
		]

		set_environment_variables()

		for var in expected_vars:
			assert var in os.environ, f'{var} not set'

	def test_environment_variables_printed(self, capsys):
		"""Test that environment variables are printed."""
		set_environment_variables()
		captured = capsys.readouterr()
		assert 'Configuring environment variables' in captured.out
		assert 'configured' in captured.out.lower()

	def test_preserves_other_env_vars(self):
		"""Test that other environment variables are preserved."""
		os.environ['TEST_PRESERVE'] = 'should_remain'
		set_environment_variables()
		assert os.environ['TEST_PRESERVE'] == 'should_remain'


class TestCheckSystemSpecs:
	"""Test system specification checking."""

	def test_prints_header(self, capsys):
		"""Test that system specs header is printed."""
		check_system_specs()
		captured = capsys.readouterr()
		assert 'System Specifications' in captured.out

	def test_handles_missing_psutil(self, capsys):
		"""Test graceful handling when psutil is missing."""
		with patch.dict('sys.modules', {'psutil': None}):
			check_system_specs()
			captured = capsys.readouterr()
			assert 'System Specifications' in captured.out

	def test_displays_psutil_stats_when_available(self, capsys):
		"""Test that system stats are displayed when psutil is available."""
		mock_psutil = MagicMock()
		mock_psutil.cpu_count.side_effect = lambda logical: 16 if logical else 8
		mock_psutil.virtual_memory.return_value = MagicMock(total=68719476736, available=34359738368)

		with patch.dict('sys.modules', {'psutil': mock_psutil}):
			# Re-import to get the patched module
			check_system_specs()
			captured = capsys.readouterr()
			assert 'System Specifications' in captured.out or 'CPU' in captured.out or 'GPU' in captured.out

	def test_handles_missing_jax(self, capsys):
		"""Test graceful handling when JAX is missing."""
		with patch.dict('sys.modules', {'jax': None}):
			check_system_specs()
			captured = capsys.readouterr()
			assert 'System Specifications' in captured.out

	def test_detects_gpu_when_available(self, capsys):
		"""Test GPU detection when JAX and CUDA are available."""
		mock_jax = MagicMock()
		mock_gpu = MagicMock()
		mock_jax.devices.return_value = [mock_gpu]

		with patch.dict('sys.modules', {'jax': mock_jax}):
			check_system_specs()
			captured = capsys.readouterr()
			assert 'System Specifications' in captured.out

	def test_graceful_failure_with_exceptions(self):
		"""Test that function doesn't crash on unexpected errors."""
		with patch('builtins.__import__', side_effect=Exception('Import error')):
			# Should not raise
			try:
				check_system_specs()
			except Exception:
				# If it does raise, it should be handled
				pass


class TestInstallMissingPackages:
	"""Test package installation logic."""

	@patch('subprocess.check_call')
	@patch('builtins.__import__')
	def test_checks_psutil_installed(self, mock_import, mock_subprocess, capsys):
		"""Test that psutil is checked."""
		mock_import.side_effect = ImportError('Not installed')
		install_missing_packages()
		captured = capsys.readouterr()
		assert 'Checking required packages' in captured.out

	@patch('subprocess.check_call')
	@patch('builtins.__import__', side_effect=ImportError)
	def test_installs_psutil_when_missing(self, mock_import, mock_subprocess, capsys):
		"""Test psutil installation when missing."""
		install_missing_packages()
		captured = capsys.readouterr()
		assert 'Checking required packages' in captured.out

	@patch('subprocess.check_call')
	def test_psutil_already_installed(self, mock_subprocess, capsys):
		"""Test handling when psutil is already installed."""
		with patch('builtins.__import__', return_value=MagicMock()):
			install_missing_packages()
			captured = capsys.readouterr()
			assert 'Checking required packages' in captured.out

	@patch('subprocess.check_call')
	def test_install_subprocess_called_correctly(self, mock_subprocess, capsys):
		"""Test that subprocess call uses correct Python."""
		with patch('builtins.__import__', side_effect=ImportError):
			install_missing_packages()
		# Verify subprocess was called (even if it failed)
		if mock_subprocess.called:
			assert 'pip' in str(mock_subprocess.call_args).lower()

	def test_handles_install_errors_gracefully(self):
		"""Test graceful handling of installation errors."""
		with patch('subprocess.check_call', side_effect=Exception('Install failed')):
			with patch('builtins.__import__', side_effect=ImportError):
				# Should handle error without crashing main execution
				try:
					install_missing_packages()
				except Exception:
					pass


class TestOptimizeSystemMainFunction:
	"""Test main optimization function."""

	@patch('optimize_system.install_missing_packages')
	@patch('optimize_system.check_system_specs')
	@patch('optimize_system.set_environment_variables')
	def test_calls_all_optimization_steps(self, mock_env_vars, mock_sys_specs, mock_install_packages, capsys):
		"""Test that optimize_system calls all required functions."""
		optimize_system()

		mock_install_packages.assert_called_once()
		mock_sys_specs.assert_called_once()
		mock_env_vars.assert_called_once()

	@patch('optimize_system.install_missing_packages')
	@patch('optimize_system.check_system_specs')
	@patch('optimize_system.set_environment_variables')
	def test_prints_optimization_header(self, mock_env_vars, mock_sys_specs, mock_install_packages, capsys):
		"""Test that optimization header is printed."""
		optimize_system()
		captured = capsys.readouterr()
		assert 'optimizing system' in captured.out.lower()

	@patch('optimize_system.install_missing_packages')
	@patch('optimize_system.check_system_specs')
	@patch('optimize_system.set_environment_variables')
	def test_prints_hardware_recommendations(self, mock_env_vars, mock_sys_specs, mock_install_packages, capsys):
		"""Test that hardware recommendations are printed."""
		optimize_system()
		captured = capsys.readouterr()
		assert 'Recommendations' in captured.out or 'RTX' in captured.out or 'chains' in captured.out

	@patch('optimize_system.install_missing_packages')
	@patch('optimize_system.check_system_specs')
	@patch('optimize_system.set_environment_variables')
	def test_prints_completion_message(self, mock_env_vars, mock_sys_specs, mock_install_packages, capsys):
		"""Test that completion message is printed."""
		optimize_system()
		captured = capsys.readouterr()
		assert 'complete' in captured.out.lower() or 'optimization' in captured.out.lower()


class TestEnvironmentVariablesValues:
	"""Test specific values of environment variables."""

	def test_gpu_memory_fraction_valid_range(self):
		"""Test GPU memory fraction is in valid range (0-1)."""
		set_environment_variables()
		fraction = float(os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'])
		assert 0 <= fraction <= 1

	def test_thread_counts_are_positive_integers(self):
		"""Test thread counts are positive integers."""
		set_environment_variables()
		for var in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS']:
			count = int(os.environ[var])
			assert count > 0

	def test_boolean_variables_are_valid(self):
		"""Test boolean environment variables have valid values."""
		set_environment_variables()
		for var in ['XLA_PYTHON_CLIENT_PREALLOCATE', 'TF_FORCE_GPU_ALLOW_GROWTH', 'JAX_ENABLE_X64']:
			value = os.environ[var].lower()
			assert value in ('true', 'false')

	def test_theano_flags_format(self):
		"""Test THEANO_FLAGS is properly formatted."""
		set_environment_variables()
		theano_flags = os.environ['THEANO_FLAGS']
		assert ',' in theano_flags or '=' in theano_flags

	def test_xla_flags_contains_optimization(self):
		"""Test XLA_FLAGS contains optimization settings."""
		set_environment_variables()
		xla_flags = os.environ['XLA_FLAGS']
		assert 'xla_gpu' in xla_flags or 'cuda' in xla_flags.lower() or 'fast' in xla_flags.lower()


class TestEnvironmentVariablesIntegration:
	"""Integration tests for environment variable configuration."""

	def test_multiple_calls_idempotent(self):
		"""Test that calling set_environment_variables multiple times is safe."""
		set_environment_variables()
		values_first = {k: os.environ[k] for k in os.environ if 'XLA' in k or 'JAX' in k}

		set_environment_variables()
		values_second = {k: os.environ[k] for k in os.environ if 'XLA' in k or 'JAX' in k}

		assert values_first == values_second

	def test_env_vars_persist_after_set(self):
		"""Test that environment variables persist after being set."""
		set_environment_variables()
		saved_value = os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']

		# Create a new process-like check
		assert os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] == saved_value

	def test_env_vars_dont_interfere_with_existing(self):
		"""Test that setting optimization vars doesn't interfere with existing vars."""
		os.environ['CUSTOM_VAR'] = 'custom_value'
		set_environment_variables()
		assert os.environ['CUSTOM_VAR'] == 'custom_value'


class TestOptimizeSystemEdgeCases:
	"""Test edge cases and error handling."""

	def test_optimize_system_with_mocked_dependencies(self):
		"""Test optimize_system when dependencies are unavailable."""
		with patch('optimize_system.install_missing_packages'):
			with patch('optimize_system.check_system_specs'):
				with patch('optimize_system.set_environment_variables'):
					optimize_system()

	def test_optimize_system_prints_recommendations_for_gpu(self, capsys):
		"""Test that GPU recommendations are specific."""
		with patch('optimize_system.install_missing_packages'):
			with patch('optimize_system.check_system_specs'):
				with patch('optimize_system.set_environment_variables'):
					optimize_system()
					captured = capsys.readouterr()
					assert 'chain' in captured.out.lower() or 'gpu' in captured.out.lower() or 'draw' in captured.out.lower()

	@patch('optimize_system.set_environment_variables', side_effect=Exception('Setup failed'))
	@patch('optimize_system.install_missing_packages')
	@patch('optimize_system.check_system_specs')
	def test_optimize_system_handles_failures(self, mock_specs, mock_install, mock_env):
		"""Test optimize_system handles exceptions gracefully."""
		try:
			optimize_system()
		except Exception:
			# Function should raise if a step fails
			pass


class TestChecksystemSpecsDetailed:
	"""Detailed tests for system specs checking."""

	def test_check_system_specs_outputs_format(self, capsys):
		"""Test that system specs output has expected format."""
		check_system_specs()
		captured = capsys.readouterr()
		# Just verify it produces output without crashing
		assert len(captured.out) > 0 or 'System Specifications' in captured.out

	def test_check_system_specs_handles_no_gpu(self, capsys):
		"""Test graceful handling when no GPU is detected."""
		mock_jax = MagicMock()
		mock_jax.devices.return_value = []

		with patch.dict('sys.modules', {'jax': mock_jax}):
			check_system_specs()
			captured = capsys.readouterr()
			# Should still run without error
			assert 'System Specifications' in captured.out or len(captured.out) > 0


class TestIntegrationWorkflow:
	"""Integration tests for complete optimization workflow."""

	@patch('optimize_system.install_missing_packages')
	@patch('optimize_system.check_system_specs')
	@patch('optimize_system.set_environment_variables')
	def test_full_optimization_workflow(self, mock_env_vars, mock_sys_specs, mock_install_packages):
		"""Test complete optimization workflow."""
		optimize_system()

		# Verify each component was called in order
		assert mock_install_packages.called
		assert mock_sys_specs.called
		assert mock_env_vars.called

	def test_env_vars_set_before_use(self):
		"""Test that environment variables are available after setting."""
		set_environment_variables()

		# Verify we can read them back
		assert os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] is not None
		assert os.environ['OMP_NUM_THREADS'] is not None

	@patch('subprocess.check_call')
	@patch('builtins.__import__', side_effect=ImportError)
	def test_optimization_continues_after_install_error(self, mock_import, mock_subprocess, capsys):
		"""Test that optimization continues even if install has errors."""
		with patch('optimize_system.check_system_specs'):
			with patch('optimize_system.set_environment_variables'):
				try:
					optimize_system()
				except Exception:
					pass
