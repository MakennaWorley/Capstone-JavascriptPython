from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_functions import coerce_dosage_classes, ensure_dir, load_meta, save_common_meta, standardize_apply, standardize_fit
from pycm import ConfusionMatrix
from torch.utils.data import DataLoader, TensorDataset

# Configure GPU acceleration for PyTorch
try:
	if torch.cuda.is_available():
		print(f'🚀 CUDA GPU available: {torch.cuda.get_device_name(0)}')
		print(f'   CUDA version: {torch.version.cuda}')
		print(f'   Number of GPUs: {torch.cuda.device_count()}')
		DEVICE = torch.device('cuda')
		GPU_AVAILABLE = True
	elif torch.backends.mps.is_available():
		print('🚀 Apple Metal GPU available')
		DEVICE = torch.device('mps')
		GPU_AVAILABLE = True
	else:
		print('💻 No GPU available, using CPU')
		DEVICE = torch.device('cpu')
		GPU_AVAILABLE = False
except Exception as e:
	print(f'GPU detection warning: {e}')
	DEVICE = torch.device('cpu')
	GPU_AVAILABLE = False


class GeneticDosageNN(nn.Module):
	"""
	Neural Network architecture for genetic dosage classification.

	Architecture:
	- Input layer with batch normalization
	- Multiple hidden layers with dropout and batch normalization
	- Output layer with 3 units (for dosage classes 0, 1, 2)
	- Uses ReLU activations and residual connections
	"""

	def __init__(
		self,
		input_dim: int,
		hidden_dims: Tuple[int, ...] = (256, 128, 64),
		dropout_rate: float = 0.3,
		use_batch_norm: bool = True,
		use_residual: bool = True,
	):
		"""
		Initialize Neural Network architecture.

		Args:
			input_dim: Number of input features
			hidden_dims: Tuple of hidden layer dimensions
			dropout_rate: Dropout probability
			use_batch_norm: Whether to use batch normalization
			use_residual: Whether to use residual connections
		"""
		super(GeneticDosageNN, self).__init__()

		self.input_dim = input_dim
		self.hidden_dims = hidden_dims
		self.dropout_rate = dropout_rate
		self.use_batch_norm = use_batch_norm
		self.use_residual = use_residual

		# Build network layers
		layers = []
		prev_dim = input_dim

		# Input batch normalization
		if use_batch_norm:
			layers.append(nn.BatchNorm1d(input_dim))

		# Hidden layers
		for i, hidden_dim in enumerate(hidden_dims):
			# Linear layer
			layers.append(nn.Linear(prev_dim, hidden_dim))

			# Batch normalization
			if use_batch_norm:
				layers.append(nn.BatchNorm1d(hidden_dim))

			# Activation
			layers.append(nn.ReLU())

			# Dropout
			if dropout_rate > 0:
				layers.append(nn.Dropout(dropout_rate))

			prev_dim = hidden_dim

		# Output layer (3 classes for dosage 0, 1, 2)
		layers.append(nn.Linear(prev_dim, 3))

		self.network = nn.Sequential(*layers)

		# Residual connections (if dimensions match)
		self.residual_layers = nn.ModuleList()
		if use_residual:
			prev_dim = input_dim
			for hidden_dim in hidden_dims:
				if prev_dim == hidden_dim:
					self.residual_layers.append(nn.Identity())
				else:
					self.residual_layers.append(nn.Linear(prev_dim, hidden_dim))
				prev_dim = hidden_dim

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Forward pass through the network.

		Args:
			x: Input tensor (batch_size, input_dim)

		Returns:
			logits: Output logits (batch_size, 3)
		"""
		return self.network(x)


class DNNDosageClassifier:
	"""
	Deep Neural Network (PyTorch-based) for genetic dosage classification.

	This model uses a deep neural network with modern architectural components:
	- Batch normalization for stable training
	- Dropout for regularization
	- Residual connections for better gradient flow
	- Support for hierarchical group structure
	- GPU acceleration via PyTorch
	- Comprehensive metrics via PYCM

	Returns:
	  - predict_proba(X): (n, 3) probability distribution over dosage classes
	  - predict_class(X): (n,) most likely dosage class
	  - predict(X): expected dosage E[y] for regression metric compatibility
	"""

	def __init__(
		self,
		*,
		hidden_dims: Tuple[int, ...] = (256, 128, 64),
		dropout_rate: float = 0.3,
		learning_rate: float = 0.001,
		batch_size: int = 256,
		epochs: int = 100,
		early_stopping_patience: int = 10,
		weight_decay: float = 1e-4,
		use_batch_norm: bool = True,
		use_residual: bool = True,
		random_seed: int = 123,
		use_gpu: bool = True,
		verbose: bool = True,
		use_class_weights: bool = True,
	):
		"""
		Initialize DNN Dosage Classifier.

		Args:
			hidden_dims: Tuple of hidden layer dimensions
			dropout_rate: Dropout probability
			learning_rate: Learning rate for optimizer
			batch_size: Batch size for training
			epochs: Maximum number of training epochs
			early_stopping_patience: Patience for early stopping
			weight_decay: L2 regularization strength
			use_batch_norm: Whether to use batch normalization
			use_residual: Whether to use residual connections
			random_seed: Random seed for reproducibility
			use_gpu: Whether to use GPU acceleration (if available)
			verbose: Print training progress
			use_class_weights: Whether to use class weights for imbalanced data
		"""
		self.hidden_dims = hidden_dims
		self.dropout_rate = dropout_rate
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.epochs = epochs
		self.early_stopping_patience = early_stopping_patience
		self.weight_decay = weight_decay
		self.use_batch_norm = use_batch_norm
		self.use_residual = use_residual
		self.random_seed = random_seed
		self.use_gpu = use_gpu and GPU_AVAILABLE
		self.verbose = verbose
		self.use_class_weights = use_class_weights

		# Set random seeds for reproducibility
		torch.manual_seed(random_seed)
		np.random.seed(random_seed)
		if torch.cuda.is_available():
			torch.cuda.manual_seed(random_seed)
			torch.cuda.manual_seed_all(random_seed)

		# Model components
		self.model: Optional[GeneticDosageNN] = None
		self.device = DEVICE if self.use_gpu else torch.device('cpu')
		self.feature_mean_: Optional[np.ndarray] = None
		self.feature_std_: Optional[np.ndarray] = None
		self.class_weights_: Optional[torch.Tensor] = None

		# Training history
		self.train_history_: Dict[str, list] = {'loss': [], 'accuracy': []}
		self.val_history_: Dict[str, list] = {'loss': [], 'accuracy': []}

		# PYCM confusion matrix for comprehensive metrics
		self.pycm_train_: Optional[ConfusionMatrix] = None
		self.pycm_metrics_: Optional[Dict[str, Any]] = None

		if self.use_gpu:
			print(f'🚀 DNN GPU acceleration enabled on {self.device}')
		else:
			print('💻 DNN running on CPU')

	@property
	def tag(self) -> str:
		return 'dnn_dosage'

	def _compute_class_weights(self, y: np.ndarray) -> torch.Tensor:
		"""
		Compute class weights for imbalanced datasets.

		Args:
			y: Class labels

		Returns:
			weights: Class weights tensor
		"""
		unique_classes, counts = np.unique(y, return_counts=True)
		total = len(y)
		weights = total / (len(unique_classes) * counts)

		# Ensure all 3 classes are represented
		weight_tensor = torch.ones(3, dtype=torch.float32)
		for cls, weight in zip(unique_classes, weights):
			weight_tensor[int(cls)] = weight

		return weight_tensor.to(self.device)

	def fit(self, X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray] = None) -> 'DNNDosageClassifier':
		"""
		Fit DNN to genetic dosage data.

		Args:
			X: Feature matrix (n_samples, n_features)
			y: Dosage labels (n_samples,) with values in {0, 1, 2}
			groups: Optional group indices (currently not used in training, but kept for API consistency)

		Returns:
			self: Fitted model
		"""
		X = np.asarray(X, dtype=np.float32)
		y_int = coerce_dosage_classes(np.asarray(y, dtype=np.float32))

		# Standardize features
		Xz, mu, sd = standardize_fit(X)
		self.feature_mean_ = mu
		self.feature_std_ = sd

		if self.verbose:
			print('\n=== Training DNN Dosage Classifier ===')
			print(f'Samples: {X.shape[0]}, Features: {X.shape[1]}')
			print(f'Dosage distribution: {np.bincount(y_int, minlength=3)}')
			print(f'Device: {self.device}')

		# Compute class weights if requested
		if self.use_class_weights:
			self.class_weights_ = self._compute_class_weights(y_int)
			if self.verbose:
				print(f'Class weights: {self.class_weights_.cpu().numpy()}')

		# Split into train and validation sets (80/20 split)
		n_samples = len(X)
		indices = np.random.permutation(n_samples)
		split_idx = int(0.8 * n_samples)
		train_idx = indices[:split_idx]
		val_idx = indices[split_idx:]

		X_train, y_train = Xz[train_idx], y_int[train_idx]
		X_val, y_val = Xz[val_idx], y_int[val_idx]

		# Create data loaders
		train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
		val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))

		train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
		val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

		# Initialize model
		self.model = GeneticDosageNN(
			input_dim=X.shape[1],
			hidden_dims=self.hidden_dims,
			dropout_rate=self.dropout_rate,
			use_batch_norm=self.use_batch_norm,
			use_residual=self.use_residual,
		).to(self.device)

		if self.verbose:
			n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
			print(f'Model parameters: {n_params:,}')
			print(f'Architecture: {self.hidden_dims}')

		# Loss function and optimizer
		if self.use_class_weights:
			criterion = nn.CrossEntropyLoss(weight=self.class_weights_)
		else:
			criterion = nn.CrossEntropyLoss()

		optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

		# Learning rate scheduler
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

		# Training loop
		best_val_loss = float('inf')
		patience_counter = 0

		# Using Checkpoints for training
		checkpoint_dir = Path(os.environ['MODELS_DIR'])
		latest_checkpoint = None
		start_epoch = 0

		# Find the highest numbered checkpoint
		checkpoints = list(checkpoint_dir.glob('checkpoint_epoch_*.pt'))
		if checkpoints:
			checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
			latest_checkpoint = checkpoints[-1]
			start_epoch = int(latest_checkpoint.stem.split('_')[-1])

		# Try to find a checkpoint if available (for resuming training)
		if latest_checkpoint and not hasattr(self, '_resumed'):
			try:
				self.model.load_state_dict(torch.load(latest_checkpoint, map_location=self.device))
				self._resumed = True
				print(f'📦 Resumed GNN from checkpoint: {latest_checkpoint.name}')
			except Exception as e:
				print(f'⚠️ Warning: Could not load checkpoint {latest_checkpoint.name}: {e}. Starting from scratch.')
				start_epoch = 0

		for epoch in range(start_epoch, self.epochs):
			# Training phase
			self.model.train()
			train_loss = 0.0
			train_correct = 0
			train_total = 0

			for batch_X, batch_y in train_loader:
				batch_X = batch_X.to(self.device)
				batch_y = batch_y.to(self.device)

				optimizer.zero_grad()
				outputs = self.model(batch_X)
				loss = criterion(outputs, batch_y)
				loss.backward()
				optimizer.step()

				train_loss += loss.item() * len(batch_X)
				_, predicted = torch.max(outputs, 1)
				train_total += batch_y.size(0)
				train_correct += (predicted == batch_y).sum().item()

			train_loss /= train_total
			train_accuracy = train_correct / train_total

			# Validation phase
			self.model.eval()
			val_loss = 0.0
			val_correct = 0
			val_total = 0

			with torch.no_grad():
				for batch_X, batch_y in val_loader:
					batch_X = batch_X.to(self.device)
					batch_y = batch_y.to(self.device)

					outputs = self.model(batch_X)
					loss = criterion(outputs, batch_y)

					val_loss += loss.item() * len(batch_X)
					_, predicted = torch.max(outputs, 1)
					val_total += batch_y.size(0)
					val_correct += (predicted == batch_y).sum().item()

			val_loss /= val_total
			val_accuracy = val_correct / val_total

			# Update history
			self.train_history_['loss'].append(train_loss)
			self.train_history_['accuracy'].append(train_accuracy)
			self.val_history_['loss'].append(val_loss)
			self.val_history_['accuracy'].append(val_accuracy)

			# Learning rate scheduling
			scheduler.step(val_loss)

			if self.verbose and (epoch + 1) % 10 == 0:
				print(
					f'Epoch [{epoch + 1}/{self.epochs}] '
					f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f} | '
					f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}'
				)

			# Early stopping
			if val_loss < best_val_loss:
				best_val_loss = val_loss
				patience_counter = 0
				# Save best model state
				self.best_model_state_ = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
			else:
				patience_counter += 1
				if patience_counter >= self.early_stopping_patience:
					if self.verbose:
						print(f'Early stopping at epoch {epoch + 1}')
					break

			# checkpoint saving
			if (epoch + 1) % 10 == 0:
				temp_path = Path(f'{os.environ["MODELS_DIR"]}/checkpoint_epoch_{epoch + 1}.pt')
				torch.save(self.model.state_dict(), temp_path)

		# Restore best model
		if hasattr(self, 'best_model_state_'):
			self.model.load_state_dict({k: v.to(self.device) for k, v in self.best_model_state_.items()})

		# Generate training predictions and compute PYCM metrics
		if self.verbose:
			print('\nComputing training metrics with PYCM...')

		y_pred_train = self.predict_class(X, groups=groups)
		self.pycm_train_ = ConfusionMatrix(actual_vector=y_int.tolist(), predict_vector=y_pred_train.tolist(), digit=4)

		# Helper function to safely convert PYCM metrics
		def safe_metric(value):
			"""Convert PYCM metric to float, handling 'None' strings and None values."""
			if value is None or value == 'None':
				return 0.0
			try:
				return float(value)
			except (ValueError, TypeError):
				return 0.0

		# Store key metrics for later access
		self.pycm_metrics_ = {
			'overall_accuracy': safe_metric(self.pycm_train_.Overall_ACC),
			'kappa': safe_metric(self.pycm_train_.Kappa),
			'overall_f1': safe_metric(self.pycm_train_.F1_Macro),
			'overall_precision': safe_metric(self.pycm_train_.PPV_Macro),
			'overall_recall': safe_metric(self.pycm_train_.TPR_Macro),
			'class_accuracy': {k: safe_metric(v) for k, v in self.pycm_train_.ACC.items()},
			'class_f1': {k: safe_metric(v) for k, v in self.pycm_train_.F1.items()},
			'class_precision': {k: safe_metric(v) for k, v in self.pycm_train_.PPV.items()},
			'class_recall': {k: safe_metric(v) for k, v in self.pycm_train_.TPR.items()},
		}

		if self.verbose:
			print('\n=== Training Metrics (PYCM) ===')
			print(f'Overall Accuracy: {self.pycm_metrics_["overall_accuracy"]:.4f}')
			print(f'Kappa: {self.pycm_metrics_["kappa"]:.4f}')
			print(f'Macro F1-Score: {self.pycm_metrics_["overall_f1"]:.4f}')
			print(f'Macro Precision: {self.pycm_metrics_["overall_precision"]:.4f}')
			print(f'Macro Recall: {self.pycm_metrics_["overall_recall"]:.4f}')
			print('\nPer-Class F1-Scores:')
			for cls, f1 in self.pycm_metrics_['class_f1'].items():
				print(f'  Dosage {cls}: {f1:.4f}')

		return self

	def predict_proba(self, X: np.ndarray, groups: Optional[np.ndarray] = None) -> np.ndarray:
		"""
		Predict probability distribution over dosage classes.

		Args:
			X: Feature matrix (n_samples, n_features)
			groups: Optional group indices (kept for API consistency)

		Returns:
			proba: Probability matrix (n_samples, 3)
		"""
		if self.model is None:
			raise RuntimeError('Model must be fitted before prediction')

		Xz = standardize_apply(X, self.feature_mean_, self.feature_std_)

		self.model.eval()
		with torch.no_grad():
			X_tensor = torch.tensor(Xz, dtype=torch.float32).to(self.device)

			# Process in batches to avoid memory issues
			batch_size = 1024
			all_probs = []

			for i in range(0, len(X_tensor), batch_size):
				batch = X_tensor[i : i + batch_size]
				logits = self.model(batch)
				probs = F.softmax(logits, dim=1)
				all_probs.append(probs.cpu().numpy())

			return np.vstack(all_probs).astype(np.float32)

	def predict_class(self, X: np.ndarray, groups: Optional[np.ndarray] = None) -> np.ndarray:
		"""
		Predict most likely dosage class.

		Args:
			X: Feature matrix (n_samples, n_features)
			groups: Optional group indices

		Returns:
			classes: Predicted dosage classes (n_samples,)
		"""
		probs = self.predict_proba(X, groups=groups)
		return np.argmax(probs, axis=1).astype(np.int64)

	def predict(self, X: np.ndarray, groups: Optional[np.ndarray] = None) -> np.ndarray:
		"""
		Expected dosage E[y] = sum_c c * p(c).
		This makes it compatible with regression metrics/plots.

		Args:
			X: Feature matrix (n_samples, n_features)
			groups: Optional group indices

		Returns:
			expected_dosage: Expected dosage values (n_samples,)
		"""
		probs = self.predict_proba(X, groups=groups)
		classes = np.array([0.0, 1.0, 2.0], dtype=np.float32)
		return (probs * classes[None, :]).sum(axis=1)

	def evaluate_pycm(self, X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray] = None) -> ConfusionMatrix:
		"""
		Evaluate model on test data using PYCM for comprehensive metrics.

		Args:
			X: Feature matrix
			y: True dosage labels
			groups: Optional group indices

		Returns:
			pycm_cm: PYCM ConfusionMatrix object with detailed metrics
		"""
		y_int = coerce_dosage_classes(np.asarray(y, dtype=np.float32))
		y_pred = self.predict_class(X, groups=groups)

		pycm_cm = ConfusionMatrix(actual_vector=y_int.tolist(), predict_vector=y_pred.tolist(), digit=4)

		return pycm_cm

	def print_pycm_report(self, pycm_cm: ConfusionMatrix, title: str = 'DNN Evaluation Report'):
		"""
		Print a comprehensive PYCM evaluation report.

		Args:
			pycm_cm: PYCM ConfusionMatrix object
			title: Report title
		"""

		# Helper function to safely format PYCM metrics
		def safe_format(value):
			"""Convert PYCM metric to float for formatting, handling 'None' strings."""
			if value is None or value == 'None':
				return 0.0
			try:
				return float(value)
			except (ValueError, TypeError):
				return 0.0

		print(f'\n{"=" * 60}')
		print(f'{title:^60}')
		print(f'{"=" * 60}')

		print('\n=== Overall Metrics ===')
		print(f'Overall Accuracy: {safe_format(pycm_cm.Overall_ACC):.4f}')
		print(f'Balanced Accuracy: {safe_format(pycm_cm.TPR_Macro):.4f}')
		print(f'Kappa: {safe_format(pycm_cm.Kappa):.4f}')
		print(f'F1-Score (Macro): {safe_format(pycm_cm.F1_Macro):.4f}')
		print(f'Precision (Macro): {safe_format(pycm_cm.PPV_Macro):.4f}')
		print(f'Recall (Macro): {safe_format(pycm_cm.TPR_Macro):.4f}')

		print('\n=== Per-Class Metrics ===')
		print(f'{"Class":<10} {"F1":<10} {"Precision":<12} {"Recall":<10} {"Accuracy":<10}')
		print('-' * 60)
		for cls in sorted(pycm_cm.classes):
			f1 = safe_format(pycm_cm.F1[cls]) if cls in pycm_cm.F1 else 0.0
			prec = safe_format(pycm_cm.PPV[cls]) if cls in pycm_cm.PPV else 0.0
			rec = safe_format(pycm_cm.TPR[cls]) if cls in pycm_cm.TPR else 0.0
			acc = safe_format(pycm_cm.ACC[cls]) if cls in pycm_cm.ACC else 0.0
			print(f'Dosage {cls:<3} {f1:<10.4f} {prec:<12.4f} {rec:<10.4f} {acc:<10.4f}')

		print('\n=== Confusion Matrix ===')
		print(pycm_cm)
		print('=' * 60)

	def save(self, paths: Dict[str, Path], extra_meta: Dict[str, Any]) -> None:
		"""
		Save DNN model to disk.

		Args:
			paths: Dictionary with file paths (dir, meta, etc.)
			extra_meta: Additional metadata to save
		"""
		if self.model is None:
			raise RuntimeError('No model to save.')

		ensure_dir(paths['dir'])

		# Save model state dict
		model_path = paths['dir'] / f'{paths["meta"].stem}.model.pt'
		torch.save(
			{
				'model_state_dict': self.model.state_dict(),
				'best_model_state': self.best_model_state_ if hasattr(self, 'best_model_state_') else None,
				'class_weights': self.class_weights_.cpu() if self.class_weights_ is not None else None,
			},
			model_path,
		)

		# Save model parameters
		payload = {
			'type': 'DNNDosageClassifier',
			'tag': self.tag,
			'feature_mean': self.feature_mean_.tolist(),
			'feature_std': self.feature_std_.tolist(),
			'params': {
				'hidden_dims': list(self.hidden_dims),
				'dropout_rate': self.dropout_rate,
				'learning_rate': self.learning_rate,
				'batch_size': self.batch_size,
				'epochs': self.epochs,
				'early_stopping_patience': self.early_stopping_patience,
				'weight_decay': self.weight_decay,
				'use_batch_norm': self.use_batch_norm,
				'use_residual': self.use_residual,
				'random_seed': self.random_seed,
				'use_gpu': self.use_gpu,
				'verbose': self.verbose,
				'use_class_weights': self.use_class_weights,
			},
			'train_history': self.train_history_,
			'val_history': self.val_history_,
			'pycm_metrics': self.pycm_metrics_,
			'model_path': str(model_path),
			'extra': extra_meta,
		}

		save_common_meta(paths, payload)

		# Save PYCM confusion matrix if available
		if self.pycm_train_ is not None:
			pycm_path = paths['dir'] / f'{paths["meta"].stem}.pycm.obj'
			self.pycm_train_.save_obj(str(pycm_path))

	@classmethod
	def load(cls, paths: Dict[str, Path]) -> 'DNNDosageClassifier':
		"""
		Load DNN model from disk.

		Args:
			paths: Dictionary with file paths

		Returns:
			model: Loaded DNNDosageClassifier
		"""
		meta = load_meta(paths)

		# Convert list back to tuple for hidden_dims
		params = meta['params'].copy()
		params['hidden_dims'] = tuple(params['hidden_dims'])

		# Create model instance
		m = cls(**params)

		# Restore feature standardization parameters
		m.feature_mean_ = np.array(meta['feature_mean'], dtype=np.float32)
		m.feature_std_ = np.array(meta['feature_std'], dtype=np.float32)

		# Restore training history
		m.train_history_ = meta.get('train_history', {'loss': [], 'accuracy': []})
		m.val_history_ = meta.get('val_history', {'loss': [], 'accuracy': []})

		# Load model state
		model_path = Path(meta['model_path'])
		if not model_path.exists():
			# Try alternate path
			model_path = paths['dir'] / f'{paths["meta"].stem}.model.pt'

		checkpoint = torch.load(model_path, map_location=m.device)

		# Reconstruct model architecture
		input_dim = len(m.feature_mean_)
		m.model = GeneticDosageNN(
			input_dim=input_dim, hidden_dims=m.hidden_dims, dropout_rate=m.dropout_rate, use_batch_norm=m.use_batch_norm, use_residual=m.use_residual
		).to(m.device)

		m.model.load_state_dict(checkpoint['model_state_dict'])

		if checkpoint.get('best_model_state') is not None:
			m.best_model_state_ = checkpoint['best_model_state']

		if checkpoint.get('class_weights') is not None:
			m.class_weights_ = checkpoint['class_weights'].to(m.device)

		# Restore PYCM metrics
		m.pycm_metrics_ = meta.get('pycm_metrics')

		# Try to load PYCM confusion matrix object
		pycm_path = paths['dir'] / f'{paths["meta"].stem}.pycm.obj'
		if pycm_path.exists():
			try:
				m.pycm_train_ = ConfusionMatrix(file=open(str(pycm_path)))
			except:
				pass  # OK if we can't load it

		m.model.eval()

		return m
