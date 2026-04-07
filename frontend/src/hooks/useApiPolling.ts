import { useEffect, useState } from 'react';
import ApiService, { type Model } from '../services/apiService.js';

/**
 * Custom hook for managing dataset polling via observable
 */
export function useDatasetsPoll(apiBase: string, apiKey: string, intervalMs: number = 5000) {
	const [datasets, setDatasets] = useState<string[]>([]);
	const [error, setError] = useState<string | null>(null);
	const [isLoading, setIsLoading] = useState(false);

	useEffect(() => {
		setIsLoading(true);
		const apiService = new ApiService(apiBase, apiKey);
		const datasetsSubscription = apiService.createDatasetsPoll(intervalMs).subscribe({
			next: (data) => {
				setDatasets(data);
				setError(null);
				setIsLoading(false);
			},
			error: (err) => {
				console.error('Dataset poll subscription error:', err);
				setError(err.message || 'Failed to load datasets');
				setIsLoading(false);
			}
		});

		// Cleanup subscription on unmount
		return () => {
			datasetsSubscription.unsubscribe();
		};
	}, [apiBase, apiKey, intervalMs]);

	return { datasets, error, isLoading };
}

/**
 * Custom hook for managing model polling via observable
 */
export function useModelsPoll(apiBase: string, apiKey: string, intervalMs: number = 5000) {
	const [models, setModels] = useState<Model[]>([]);
	const [error, setError] = useState<string | null>(null);
	const [isLoading, setIsLoading] = useState(false);

	useEffect(() => {
		setIsLoading(true);
		const apiService = new ApiService(apiBase, apiKey);
		const modelsSubscription = apiService.createModelsPoll(intervalMs).subscribe({
			next: (data) => {
				setModels(data);
				setError(null);
				setIsLoading(false);
			},
			error: (err) => {
				console.error('Models poll subscription error:', err);
				setError(err.message || 'Failed to load models');
				setIsLoading(false);
			}
		});

		// Cleanup subscription on unmount
		return () => {
			modelsSubscription.unsubscribe();
		};
	}, [apiBase, apiKey, intervalMs]);

	return { models, error, isLoading };
}
