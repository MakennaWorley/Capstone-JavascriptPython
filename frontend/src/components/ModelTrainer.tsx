import { useState } from 'react';
import LoadingProgress from './LoadingProgress.js';

type Model = {
	model_name: string;
	model_type: string;
};

type ModelTrainerProps = {
	apiBase: string;
	xApiKey: string;
	selectedDataset: string;
	selectedModel: Model | null;
	onTestComplete?: (data: { paths: any; testMetrics: any; images: any }) => void;
};

type ApiSuccessTest = {
	status: 'success';
	message: string;
	data: {
		test_metrics: any;
		paths: {
			graph_test: string;
			graph_cm: string;
			model_dir: string;
		};
		images: {
			graph_test_base64?: string;
			graph_cm_base64?: string;
		};
	};
};

type ApiError = {
	status: 'error';
	code?: string;
	message: string;
};

export default function ModelTrainer({ apiBase, xApiKey, selectedDataset, selectedModel, onTestComplete }: ModelTrainerProps) {
	const [loading, setLoading] = useState(false);
	const [error, setError] = useState<string | null>(null);

	const canTest = selectedDataset && selectedModel;

	async function handleTestModel() {
		if (!canTest) return;

		setLoading(true);
		setError(null);

		try {
			const response = await fetch(`${apiBase}/api/models/test`, {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json',
					'x-api-key': xApiKey
				},
				body: JSON.stringify({
					dataset_name: selectedDataset,
					model_name: selectedModel.model_name,
					model_type: selectedModel.model_type
				})
			});

			const result: ApiSuccessTest | ApiError = await response.json();

			if (result.status === 'success') {
				setError(null);
				if (onTestComplete) {
					onTestComplete({
						paths: result.data.paths,
						testMetrics: result.data.test_metrics,
						images: result.data.images
					});
				}
			} else {
				// Display error from API response
				const errorMessage = result.message || `Failed to test model (Status: ${result.status})`;
				setError(errorMessage);
				console.error('Model test error:', result);
			}
		} catch (err) {
			const errorMsg = err instanceof Error ? err.message : String(err);
			setError(`Network error: ${errorMsg}`);
			console.error('Model test network error:', err);
		} finally {
			setLoading(false);
		}
	}

	return (
		<div style={{ marginTop: '2rem', padding: '1.5rem', backgroundColor: '#1a1a1a', borderRadius: '8px' }}>
			<h3 style={{ marginTop: 0 }}>Model Trainer</h3>

			<div style={{ marginBottom: '1rem' }}>
				<p style={{ marginBottom: '0.5rem' }}>
					<strong>Selected Dataset:</strong> {selectedDataset || <em style={{ opacity: 0.6 }}>None</em>}
				</p>
				<p style={{ marginBottom: '0.5rem' }}>
					<strong>Selected Model:</strong>{' '}
					{selectedModel ? (
						<>
							{selectedModel.model_name} ({selectedModel.model_type})
						</>
					) : (
						<em style={{ opacity: 0.6 }}>None</em>
					)}
				</p>
			</div>

			<button
				type="button"
				onClick={handleTestModel}
				disabled={!canTest || loading}
				style={{
					padding: '0.6rem 1.2rem',
					backgroundColor: canTest && !loading ? '#4CAF50' : '#555',
					color: 'white',
					border: 'none',
					borderRadius: '4px',
					cursor: canTest && !loading ? 'pointer' : 'not-allowed',
					fontSize: '1rem'
				}}
			>
				{loading ? 'Testing...' : 'Test Model on Dataset'}
			</button>

			<LoadingProgress isLoading={loading} message="Applying a model to your data..." />

			{!canTest && <p style={{ marginTop: '1rem', color: '#ff9800' }}>Please select both a dataset and a model to test.</p>}

			{error && (
				<div
					style={{
						marginTop: '1rem',
						padding: '1rem',
						backgroundColor: '#3d1a1a',
						borderLeft: '4px solid #f44336',
						borderRadius: '4px'
					}}
				>
					<strong>Error:</strong> {error}
				</div>
			)}
		</div>
	);
}
