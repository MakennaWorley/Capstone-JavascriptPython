import base64
import io
import os
import zipfile
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .data_generation import create_data_from_params
from .functions import (
	DashboardFilesMissing,
	api_error,
	api_success,
	get_all_dataset_files,
	get_dataset_dashboard_files,
	get_dataset_names,
	get_individual_family_tree_data,
	get_model_list,
)
from .model_main import test_on_new_data

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
load_dotenv(dotenv_path=ROOT_DIR / '.env')

app = FastAPI()

origins = os.getenv('CORS_ORIGINS', '')
origins_list = [o.strip() for o in origins.split(',') if o.strip()]

app.add_middleware(CORSMiddleware, allow_origins=origins_list, allow_credentials=True, allow_methods=['*'], allow_headers=['*'])

DATASETS_DIR = (BASE_DIR / os.getenv('DATASETS_DIR')).resolve()
PROTECTED_DATASETS_DIR = (BASE_DIR / os.getenv('PROTECTED_DATASETS_DIR')).resolve()
MODELS_DIR = (BASE_DIR / os.getenv('MODELS_DIR')).resolve()
IMAGES_DIR = (BASE_DIR / os.getenv('IMAGES_DIR')).resolve()
LOGS_DIR = (BASE_DIR / os.getenv('LOGS_DIR')).resolve()


# debugging
@app.on_event('startup')
async def show_routes():
	print('=== ROUTES ===')
	for r in app.routes:
		try:
			print(f'{getattr(r, "methods", "")} {r.path}')
		except Exception:
			pass
	print('==============')


@app.on_event('startup')
async def verify_paths():
	print(f'DEBUG: DATASETS_DIR is resolved to: {DATASETS_DIR}')
	if not DATASETS_DIR.exists():
		print('WARNING: DATASETS_DIR does not exist!')
	print(f'DEBUG: MODELS_DIR is resolved to: {MODELS_DIR}')
	if not MODELS_DIR.exists():
		print('WARNING: MODELS_DIR does not exist!')


# health check
@app.get('/api/hello')
def hello():
	return {'message': 'Hello from FastAPI'}


@app.post('/api/create/data')
async def create_dataset(request: Request):
	try:
		body = await request.json()
	except Exception:
		return api_error(message='Error: Request body must be valid JSON', status_code=400, code='INVALID_JSON')

	try:
		simulation_params = body.get('params', body)

		if not isinstance(simulation_params, dict) or len(simulation_params) == 0:
			return api_error(message='Error: Missing or invalid simulation params', status_code=400, code='INVALID_PARAMS')

		os.makedirs(DATASETS_DIR, exist_ok=True)
		simulation_params['datasets_dir'] = str(DATASETS_DIR)

		res = create_data_from_params(simulation_params)

		return api_success(
			message='Success: Data generated successfully',
			data={
				'dataset_name': simulation_params.get('name'),
				'output_dir': str(DATASETS_DIR),
				'result': {'config_used': res['config'], 'file_paths': res['outputs']},
			},
			status_code=200,
		)

	except ValueError as e:
		# Handle validation errors (e.g., dataset already exists)
		return api_error(message=f'Error: {str(e)}', status_code=400, code='VALIDATION_ERROR')
	except Exception as e:
		print(f'Error during data generation: {str(e)}')
		return api_error(message='Unexpected server error while generating data', status_code=500, code='DATASET_CREATE_FAILED')


@app.get('/api/datasets/list', response_model=List[str])
async def list_datasets():
	try:
		dataset_names = get_dataset_names()

		return api_success(
			message='Success: Datasets retrieved successfully', data={'datasets': dataset_names, 'count': len(dataset_names)}, status_code=200
		)

	except UnicodeDecodeError:
		return api_error(message='Error: Could not find dataset list', status_code=500, code='DATASET_FILE_MISSING')

	except Exception as e:
		print(f'Error: Could not read dataset list: {str(e)}')
		return api_error(message='Unexpected server error while reading datasets', status_code=500, code='DATASET_LIST_FAILED')


@app.get('/api/dataset/{dataset_name}/dashboard')
async def dataset_dashboard(dataset_name: str):
	try:
		data = get_dataset_dashboard_files(dataset_name, datasets_dir=DATASETS_DIR)

		return api_success(message=f"Success: Dashboard files returned for dataset '{dataset_name}'", data=data, status_code=200)

	except DashboardFilesMissing as e:
		return api_error(
			message=f"Missing required files for dataset '{e.dataset_name}': {', '.join(e.missing)}", status_code=404, code='DASHBOARD_FILES_MISSING'
		)

	except UnicodeDecodeError:
		return api_error(message='Error: Could not decode one of the CSV files (encoding issue)', status_code=500, code='CSV_DECODE_FAILED')

	except Exception as e:
		print(f'Error: Dashboard fetch failed: {str(e)}')
		return api_error(message='Unexpected server error while building dashboard response', status_code=500, code='DASHBOARD_FAILED')


@app.get('/api/dataset/{dataset_name}/tree/{individual_id}')
async def dataset_family_tree(dataset_name: str, individual_id: int):
	"""
	Fetch the connected family tree and genetic data for a specific individual.
	"""
	print('this is the request', dataset_name, individual_id)

	try:
		data = get_individual_family_tree_data(dataset_name, individual_id, datasets_dir=DATASETS_DIR)

		return api_success(message=f'Success: Family tree retrieved for individual {individual_id}', data=data, status_code=200)

	except FileNotFoundError as e:
		return api_error(message=f'Required file not found: {str(e)}', status_code=404, code='FILE_NOT_FOUND')

	except KeyError as e:
		return api_error(message=str(e).strip("'"), status_code=404, code='INDIVIDUAL_NOT_FOUND')

	except Exception as e:
		print(f'Error: Family tree fetch failed: {str(e)}')
		return api_error(message='Unexpected server error while building family tree', status_code=500, code='TREE_BUILD_FAILED')


@app.get('/api/dataset/{dataset_name}/download')
async def download_dataset(dataset_name: str):
	"""
	Download ALL dataset files as a single zip.
	"""
	try:
		files = get_all_dataset_files(dataset_name, datasets_dir=DATASETS_DIR)

		# Build zip in-memory
		buf = io.BytesIO()
		with zipfile.ZipFile(buf, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
			# Text files
			zf.writestr(f'{dataset_name}.truth_genotypes.csv', files['truth_genotypes_csv'])
			zf.writestr(f'{dataset_name}.observed_genotypes.csv', files['observed_genotypes_csv'])
			zf.writestr(f'{dataset_name}.pedigree.csv', files['pedigree_csv'])
			zf.writestr(f'{dataset_name}.run_metadata.json', files['run_metadata_json'])

			# Binary trees (base64 -> bytes)
			trees_bytes = base64.b64decode(files['trees_base64'])
			zf.writestr(f'{dataset_name}.trees', trees_bytes)

		buf.seek(0)

		return StreamingResponse(buf, media_type='application/zip', headers={'Content-Disposition': f'attachment; filename="{dataset_name}.zip"'})

	except DashboardFilesMissing as e:
		return api_error(
			message=f"Missing required files for dataset '{e.dataset_name}': {', '.join(e.missing)}", status_code=404, code='DATASET_FILES_MISSING'
		)

	except Exception as e:
		print(f'Error: Dataset download failed: {str(e)}')
		return api_error(message='Unexpected server error while building dataset zip', status_code=500, code='DATASET_ZIP_FAILED')


@app.get('/api/models/list', response_model=List[dict])
async def list_models():
	"""
	List all trained models from models.csv.
	Returns an array of objects with model_name and model_type.
	"""
	try:
		models = get_model_list(models_dir=MODELS_DIR)

		return api_success(message='Success: Models retrieved successfully', data={'models': models, 'count': len(models)}, status_code=200)

	except FileNotFoundError:
		return api_error(message='Error: models.csv not found', status_code=404, code='MODELS_FILE_MISSING')

	except Exception as e:
		print(f'Error: Could not read models list: {str(e)}')
		return api_error(message='Unexpected server error while reading models', status_code=500, code='MODELS_LIST_FAILED')


@app.post('/api/models/test')
async def test_model_on_dataset(request: Request):
	"""
	Test a trained model on a new dataset.
	Expects JSON body with:
		- dataset_name: str (the test dataset)
		- model_name: str (the trained model name)
		- model_type: str (e.g., 'multi_log_regression' or 'bayes_softmax3')

	Returns the test log content as raw text.
	"""
	try:
		body = await request.json()
	except Exception:
		return api_error(message='Error: Request body must be valid JSON', status_code=400, code='INVALID_JSON')

	try:
		dataset_name = body.get('dataset_name')
		model_name = body.get('model_name')
		model_type = body.get('model_type')

		if not dataset_name or not model_name or not model_type:
			return api_error(message='Error: Missing required fields (dataset_name, model_name, model_type)', status_code=400, code='INVALID_PARAMS')

		# Call test_on_new_data with the appropriate directories
		result = test_on_new_data(
			test_base=dataset_name,
			model_type=model_type,
			model_name=model_name,
			models_dir=str(MODELS_DIR),
			images_dir=str(MODELS_DIR.parent / 'images'),
			datasets_dir=str(DATASETS_DIR),
		)

		# Read and encode the images as base64
		paths_data = result.get('paths', {})
		graph_test_path = Path(paths_data.get('graph_test', ''))
		graph_cm_path = Path(paths_data.get('graph_cm', ''))

		image_data = {}
		if graph_test_path.is_file():
			image_data['graph_test_base64'] = base64.b64encode(graph_test_path.read_bytes()).decode('ascii')
		if graph_cm_path.is_file():
			image_data['graph_cm_base64'] = base64.b64encode(graph_cm_path.read_bytes()).decode('ascii')

		return api_success(
			message=f"Success: Model '{model_name}' tested on dataset '{dataset_name}'",
			data={
				'test_metrics': result.get('test_metrics'),
				'paths': result.get('paths'),
				'images': image_data,
				'prediction_errors': result.get('prediction_errors', []),
			},
			status_code=200,
		)

	except FileNotFoundError as e:
		error_msg = f'Model not found - {str(e)}'
		print(f'[FileNotFoundError] {error_msg}')
		return api_error(message=error_msg, status_code=404, code='MODEL_NOT_FOUND')

	except ValueError as e:
		error_msg = str(e)
		print(f'[ValueError] {error_msg}')
		import traceback

		traceback.print_exc()
		return api_error(message=error_msg, status_code=400, code='VALIDATION_ERROR')

	except Exception as e:
		error_msg = f'Unexpected server error while testing model: {str(e)}'
		print(f'[Exception] {error_msg}')
		import traceback

		traceback.print_exc()
		return api_error(message=error_msg, status_code=500, code='MODEL_TEST_FAILED')
