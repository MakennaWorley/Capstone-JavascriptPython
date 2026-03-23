from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pycm import ConfusionMatrix

try:
	from torch_geometric.data import Data
	from torch_geometric.data import DataLoader as GeometricDataLoader
	from torch_geometric.nn import GraphConv, global_mean_pool

	TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
	TORCH_GEOMETRIC_AVAILABLE = False
	print('Warning: torch_geometric not installed. Install with: pip install torch-geometric')

from .model_functions import coerce_dosage_classes, ensure_dir, load_meta, save_common_meta, standardize_apply, standardize_fit

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


class GeneticDosageGNN(nn.Module):
	"""
	Graph Neural Network architecture for genetic dosage classification.

	Architecture:
	- Builds a graph where nodes represent genetic features/variants
	- Edges connect correlated features (similarity > threshold)
	- Multiple GraphConv layers to aggregate information across the feature graph
	- Global mean pooling to create sample-level representations
	- Output layer with 3 units (for dosage classes 0, 1, 2)
	"""

	def __init__(self, input_dim: int, hidden_dims: Tuple[int, ...] = (256, 128, 64), dropout_rate: float = 0.3, use_batch_norm: bool = True):
		"""
		Initialize GNN architecture.

		Args:
			input_dim: Number of input features (nodes in graph)
			hidden_dims: Tuple of hidden layer dimensions for GNN layers
			dropout_rate: Dropout probability
			use_batch_norm: Whether to use batch normalization
		"""
		super(GeneticDosageGNN, self).__init__()

		self.input_dim = input_dim
		self.hidden_dims = hidden_dims
		self.dropout_rate = dropout_rate
		self.use_batch_norm = use_batch_norm

		if not TORCH_GEOMETRIC_AVAILABLE:
			raise ImportError('torch_geometric is required for GNN model')

		# Build GNN layers
		self.gnn_layers = nn.ModuleList()
		self.batch_norms = nn.ModuleList()

		prev_dim = input_dim

		# GraphConv layers
		for hidden_dim in hidden_dims:
			self.gnn_layers.append(GraphConv(prev_dim, hidden_dim))
			if use_batch_norm:
				self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
			prev_dim = hidden_dim

		# Output layer
		self.output_layer = nn.Linear(prev_dim, 3)

	def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
		"""
		Forward pass through the GNN.

		Args:
			x: Node feature matrix (num_nodes, input_dim)
			edge_index: Graph edge indices (2, num_edges)
			batch: Batch assignment for each node

		Returns:
			logits: Output logits (batch_size, 3)
		"""
		for i, gnn_layer in enumerate(self.gnn_layers):
			x = gnn_layer(x, edge_index)
			if self.use_batch_norm:
				x = self.batch_norms[i](x)
				x = F.relu(x)
			if self.dropout_rate > 0:
				x = F.dropout(x, p=self.dropout_rate, training=self.training)

		# Global mean pooling to aggregate node features into graph-level features
		x = global_mean_pool(x, batch)

		# Output layer
		logits = self.output_layer(x)
		return logits


class GNNDosageClassifier:
	"""
	Graph Neural Network (PyTorch-based) for genetic dosage classification.

	This model uses Graph Convolutional Networks to leverage correlations between genetic features:
	- Builds a feature correlation graph where edges connect related variants
	- Multiple GraphConv layers to propagate information across features
	- Global mean pooling to create sample-level predictions
	- Batch normalization and dropout for regularization
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
		random_seed: int = 123,
		use_gpu: bool = True,
		verbose: bool = True,
		use_class_weights: bool = True,
		correlation_threshold: float = 0.3,
	):
		"""
		Initialize GNN Dosage Classifier.

		Args:
			hidden_dims: Tuple of hidden GNN layer dimensions
			dropout_rate: Dropout probability
			learning_rate: Learning rate for optimizer
			batch_size: Batch size for training
			epochs: Maximum number of training epochs
			early_stopping_patience: Patience for early stopping
			weight_decay: L2 regularization strength
			use_batch_norm: Whether to use batch normalization
			random_seed: Random seed for reproducibility
			use_gpu: Whether to use GPU acceleration (if available)
			verbose: Print training progress
			use_class_weights: Whether to use class weights for imbalanced data
			correlation_threshold: Threshold for feature correlation edges (0-1)
		"""
		if not TORCH_GEOMETRIC_AVAILABLE:
			raise ImportError('torch_geometric is required for GNN model. Install with: pip install torch-geometric')

		self.hidden_dims = hidden_dims
		self.dropout_rate = dropout_rate
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.epochs = epochs
		self.early_stopping_patience = early_stopping_patience
		self.weight_decay = weight_decay
		self.use_batch_norm = use_batch_norm
		self.random_seed = random_seed
		self.use_gpu = use_gpu and GPU_AVAILABLE
		self.verbose = verbose
		self.use_class_weights = use_class_weights
		self.correlation_threshold = correlation_threshold

		# Set random seeds for reproducibility
		torch.manual_seed(random_seed)
		np.random.seed(random_seed)
		if torch.cuda.is_available():
			torch.cuda.manual_seed(random_seed)
			torch.cuda.manual_seed_all(random_seed)

		# Model components
		self.model: Optional[GeneticDosageGNN] = None
		self.device = DEVICE if self.use_gpu else torch.device('cpu')
		self.feature_mean_: Optional[np.ndarray] = None
		self.feature_std_: Optional[np.ndarray] = None
		self.class_weights_: Optional[torch.Tensor] = None
		self.edge_index_: Optional[torch.Tensor] = None

		# Training history
		self.train_history_: Dict[str, list] = {'loss': [], 'accuracy': []}
		self.val_history_: Dict[str, list] = {'loss': [], 'accuracy': []}

		# PYCM confusion matrix for comprehensive metrics
		self.pycm_train_: Optional[ConfusionMatrix] = None
		self.pycm_metrics_: Optional[Dict[str, Any]] = None

		if self.use_gpu:
			print(f'🚀 GNN GPU acceleration enabled on {self.device}')
		else:
			print('💻 GNN running on CPU')

	@property
	def tag(self) -> str:
		return 'gnn_dosage'

	def _build_feature_graph(self, X: np.ndarray) -> torch.Tensor:
		"""
		Build a graph where nodes are features and edges connect correlated features.

		Args:
			X: Standardized feature matrix (n_samples, n_features)

		Returns:
			edge_index: PyTorch tensor of shape (2, num_edges)
		"""
		# Compute correlation matrix between features
		feature_corr = np.corrcoef(X.T)  # (n_features, n_features)

		# Find edges based on correlation threshold
		edges = []
		n_features = X.shape[1]

		for i in range(n_features):
			for j in range(i + 1, n_features):
				# Create edge if absolute correlation exceeds threshold
				if abs(feature_corr[i, j]) >= self.correlation_threshold:
					edges.append([i, j])
					edges.append([j, i])  # Undirected graph

		if len(edges) == 0:
			# If no edges from correlation, create k-NN graph (k=3 nearest features)
			distances = 1 - np.abs(feature_corr)
			for i in range(n_features):
				nearest = np.argsort(distances[i])[1:4]  # Skip self (index 0)
				for j in nearest:
					edges.append([i, j])

		edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

		if self.verbose:
			print(f'Built feature graph with {n_features} nodes and {edge_index.shape[1] // 2} undirected edges')

		return edge_index

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

	def fit(self, X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray] = None) -> 'GNNDosageClassifier':
		"""
		Fit GNN to genetic dosage data.

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
			print('\n=== Training GNN Dosage Classifier ===')
			print(f'Samples: {X.shape[0]}, Features: {X.shape[1]}')
			print(f'Dosage distribution: {np.bincount(y_int, minlength=3)}')
			print(f'Device: {self.device}')

		# Build feature correlation graph
		self.edge_index_ = self._build_feature_graph(Xz).to(self.device)

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

		# Create PyTorch Geometric Data objects for each sample
		train_graphs = []
		for i, (x_sample, y_sample) in enumerate(zip(X_train, y_train)):
			# Node features are the feature values
			node_features = torch.tensor(x_sample, dtype=torch.float32).unsqueeze(1)  # (n_features, 1)
			graph = Data(x=node_features, edge_index=self.edge_index_, y=torch.tensor(y_sample, dtype=torch.long))
			train_graphs.append(graph)

		val_graphs = []
		for i, (x_sample, y_sample) in enumerate(zip(X_val, y_val)):
			node_features = torch.tensor(x_sample, dtype=torch.float32).unsqueeze(1)
			graph = Data(x=node_features, edge_index=self.edge_index_, y=torch.tensor(y_sample, dtype=torch.long))
			val_graphs.append(graph)

		train_loader = GeometricDataLoader(train_graphs, batch_size=self.batch_size, shuffle=True)
		val_loader = GeometricDataLoader(val_graphs, batch_size=self.batch_size, shuffle=False)

		# Initialize model
		self.model = GeneticDosageGNN(
			input_dim=1,  # Each node feature is 1D (single feature value)
			hidden_dims=self.hidden_dims,
			dropout_rate=self.dropout_rate,
			use_batch_norm=self.use_batch_norm,
		).to(self.device)

		if self.verbose:
			n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
			print(f'Model parameters: {n_params:,}')
			print(f'Architecture: {self.hidden_dims} (GNN layers)')

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

		for epoch in range(self.epochs):
			# Training phase
			self.model.train()
			train_loss = 0.0
			train_correct = 0
			train_total = 0

			for batch in train_loader:
				batch = batch.to(self.device)

				optimizer.zero_grad()
				outputs = self.model(batch.x, batch.edge_index, batch.batch)
				loss = criterion(outputs, batch.y)
				loss.backward()
				optimizer.step()

				train_loss += loss.item() * len(batch.y)
				_, predicted = torch.max(outputs, 1)
				train_total += len(batch.y)
				train_correct += (predicted == batch.y).sum().item()

			train_loss /= train_total
			train_accuracy = train_correct / train_total

			# Validation phase
			self.model.eval()
			val_loss = 0.0
			val_correct = 0
			val_total = 0

			with torch.no_grad():
				for batch in val_loader:
					batch = batch.to(self.device)

					outputs = self.model(batch.x, batch.edge_index, batch.batch)
					loss = criterion(outputs, batch.y)

					val_loss += loss.item() * len(batch.y)
					_, predicted = torch.max(outputs, 1)
					val_total += len(batch.y)
					val_correct += (predicted == batch.y).sum().item()

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

		# Create graph objects for each sample
		test_graphs = []
		for x_sample in Xz:
			node_features = torch.tensor(x_sample, dtype=torch.float32).unsqueeze(1)
			graph = Data(x=node_features, edge_index=self.edge_index_)
			test_graphs.append(graph)

		test_loader = GeometricDataLoader(test_graphs, batch_size=self.batch_size, shuffle=False)

		self.model.eval()
		all_probs = []

		with torch.no_grad():
			for batch in test_loader:
				batch = batch.to(self.device)
				logits = self.model(batch.x, batch.edge_index, batch.batch)
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

	def print_pycm_report(self, pycm_cm: ConfusionMatrix, title: str = 'GNN Evaluation Report'):
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
		Save GNN model to disk.

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
				'edge_index': self.edge_index_.cpu() if self.edge_index_ is not None else None,
			},
			model_path,
		)

		# Save model parameters
		payload = {
			'type': 'GNNDosageClassifier',
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
				'random_seed': self.random_seed,
				'use_gpu': self.use_gpu,
				'verbose': self.verbose,
				'use_class_weights': self.use_class_weights,
				'correlation_threshold': self.correlation_threshold,
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
	def load(cls, paths: Dict[str, Path]) -> 'GNNDosageClassifier':
		"""
		Load GNN model from disk.

		Args:
			paths: Dictionary with file paths

		Returns:
			model: Loaded GNNDosageClassifier
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

		# Restore edge index
		if checkpoint.get('edge_index') is not None:
			m.edge_index_ = checkpoint['edge_index'].to(m.device)

		# Reconstruct model architecture
		m.model = GeneticDosageGNN(input_dim=1, hidden_dims=m.hidden_dims, dropout_rate=m.dropout_rate, use_batch_norm=m.use_batch_norm).to(m.device)

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
			except Exception:
				pass  # OK if we can't load it

		m.model.eval()

		return m
