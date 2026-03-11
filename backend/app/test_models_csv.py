"""
Test script to verify models.csv functionality
"""

from pathlib import Path

from model_main import update_models_csv

# Test the CSV update function
print('Testing update_models_csv function...\n')

# Create a test models directory
test_models_dir = Path('test_models_dir')
test_models_dir.mkdir(exist_ok=True)

# Clean up any existing test file
test_csv = test_models_dir / 'models.csv'
if test_csv.exists():
	test_csv.unlink()
	print(f'Removed existing {test_csv}\n')

# Test 1: Create new CSV with first entry
print('Test 1: Adding first entry')
update_models_csv('testing.training', 'bayes_softmax3', csv_path=test_csv)

# Test 2: Add another entry
print('\nTest 2: Adding second entry')
update_models_csv('testing.training', 'multi_log_regression', csv_path=test_csv)

# Test 3: Add entries for different training set
print('\nTest 3: Adding entries for different training set')
update_models_csv('bettersample.training', 'bayes_softmax3', csv_path=test_csv)
update_models_csv('bettersample.training', 'multi_log_regression', csv_path=test_csv)

# Test 4: Try to add duplicate (should skip)
print('\nTest 4: Attempting to add duplicate entry')
update_models_csv('testing.training', 'bayes_softmax3', csv_path=test_csv)

# Display the final CSV contents
print('\n' + '=' * 60)
print(f'Final {test_csv} contents:')
print('=' * 60)
if test_csv.exists():
	with open(test_csv, 'r') as f:
		print(f.read())
else:
	print('File not found!')

print(f'\nTest completed! Check {test_csv} file.')
print(f'CSV location: {test_csv.absolute()}')
