import json
import tempfile
from pathlib import Path

import numpy as np
from model_hmm import HMMDosageClassifier

# Create and fit model
model = HMMDosageClassifier(n_iter=5, verbose=False, random_seed=42)
np.random.seed(42)
X = np.random.randn(30, 5).astype(np.float32)
y = np.tile([0, 1, 2], 10)

model.fit(X, y)

print(f'After fit - shape of covars: {model.model.covars_.shape}')
print(f'After fit - covars:\n{model.model.covars_}')

with tempfile.TemporaryDirectory() as tmpdir:
	model_dir = Path(tmpdir) / 'models'
	model_dir.mkdir(parents=True)
	meta_path = model_dir / 'test_model.json'

	paths = {'dir': model_dir, 'meta': meta_path}

	# Save
	model.save(paths, extra_meta={'version': '1.0'})

	# Check what's in the JSON
	with open(meta_path) as f:
		data = json.load(f)

	print('\nSaved covars (as list):')
	print(data['model_params']['covars'])
	print(f'Type of saved covars: {type(data["model_params"]["covars"])}')
	print(f'Shape when converted to array: {np.array(data["model_params"]["covars"]).shape}')

	# What happens when we np.array it
	loaded_covars = np.array(data['model_params']['covars'])
	print(f'\nLoaded covars shape: {loaded_covars.shape}')
	print(f'Loaded covars:\n{loaded_covars}')
