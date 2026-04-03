import { Observable, interval, of } from 'rxjs';
import { catchError, startWith, switchMap } from 'rxjs/operators';

export type Model = {
	model_name: string;
	model_type: string;
};

export type ApiSuccessDatasets = {
	status: 'success';
	message: string;
	data: {
		datasets: string[];
		count: number;
	};
};

export type ApiSuccessModels = {
	status: 'success';
	message: string;
	data: {
		models: Model[];
		count: number;
	};
};

export type ApiError = {
	status: 'error';
	code?: string;
	message: string;
};

class ApiService {
	private apiBase: string;
	private apiKey: string;

	constructor(apiBase: string, apiKey: string) {
		this.apiBase = apiBase;
		this.apiKey = apiKey;
	}

	/**
	 * Fetches datasets from the backend
	 */
	private fetchDatasetsOnce(): Promise<string[]> {
		return fetch(`${this.apiBase}/api/datasets/list`, {
			method: 'GET',
			headers: { 'x-api-key': this.apiKey }
		})
			.then((r) => r.json())
			.then((j: ApiSuccessDatasets | ApiError) => {
				if (j.status === 'success') {
					return j.data.datasets;
				}
				throw new Error(j.message || 'Failed to fetch datasets');
			});
	}

	/**
	 * Fetches models from the backend
	 */
	private fetchModelsOnce(): Promise<Model[]> {
		return fetch(`${this.apiBase}/api/models/list`, {
			method: 'GET',
			headers: { 'x-api-key': this.apiKey }
		})
			.then((r) => r.json())
			.then((j: ApiSuccessModels | ApiError) => {
				if (j.status === 'success') {
					return j.data.models;
				}
				throw new Error(j.message || 'Failed to fetch models');
			});
	}

	/**
	 * Creates an observable that polls the datasets endpoint periodically
	 * @param intervalMs - How often to poll in milliseconds (default: 5000ms / 5 seconds)
	 */
	createDatasetsPoll(intervalMs: number = 5000): Observable<string[]> {
		return interval(intervalMs).pipe(
			startWith(0), // Emit immediately on subscription
			switchMap(() => {
				return new Observable<string[]>((subscriber) => {
					this.fetchDatasetsOnce()
						.then((datasets) => {
							subscriber.next(datasets);
							subscriber.complete();
						})
						.catch((error) => {
							console.error('Error fetching datasets:', error);
							subscriber.error(error);
						});
				});
			}),
			catchError((error) => {
				console.error('Dataset polling error:', error);
				// Return empty array on error to continue polling
				return of([]);
			})
		);
	}

	/**
	 * Creates an observable that polls the models endpoint periodically
	 * @param intervalMs - How often to poll in milliseconds (default: 5000ms / 5 seconds)
	 */
	createModelsPoll(intervalMs: number = 5000): Observable<Model[]> {
		return interval(intervalMs).pipe(
			startWith(0), // Emit immediately on subscription
			switchMap(() => {
				return new Observable<Model[]>((subscriber) => {
					this.fetchModelsOnce()
						.then((models) => {
							subscriber.next(models);
							subscriber.complete();
						})
						.catch((error) => {
							console.error('Error fetching models:', error);
							subscriber.error(error);
						});
				});
			}),
			catchError((error) => {
				console.error('Model polling error:', error);
				// Return empty array on error to continue polling
				return of([]);
			})
		);
	}
}

export default ApiService;
