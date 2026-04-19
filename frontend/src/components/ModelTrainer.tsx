import { Alert, Button } from '@mui/material';
import { useState } from 'react';
import LoadingProgress from './LoadingProgress.js';

const MODEL_TYPE_SHORT_NAMES: Record<string, string> = {
	bayes_softmax3: 'Bayesian Inference',
	multi_log_regression: 'Multinomial Logistic Regression',
	hmm_dosage: 'Hidden Markov Model',
	dnn_dosage: 'Deep Neural Network',
	gnn_dosage: 'Graph Neural Network'
};

function capitalize(s: string): string {
	if (!s) return s;
	return s.charAt(0).toUpperCase() + s.slice(1);
}

type Model = {
	model_name: string;
	model_type: string;
};

type ModelTrainerProps = {
	apiBase: string;
	xApiKey: string;
	selectedDataset: string;
	selectedModel: Model | null;
	onTestComplete?: (data: {
		paths: any;
		testMetrics: any;
		images: any;
		predictionErrors: Array<{ individual: string; site: string; predicted: number; actual: number }>;
	}) => void;
	nerdMode: boolean;
	onNerdModeChange: (value: boolean) => void;
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
		prediction_errors?: Array<{ individual: string; site: string; predicted: number; actual: number }>;
	};
};

type ApiError = {
	status: 'error';
	code?: string;
	message: string;
};

export default function ModelTrainer({ apiBase, xApiKey, selectedDataset, selectedModel, onTestComplete, nerdMode, onNerdModeChange }: ModelTrainerProps) {
	const [loading, setLoading] = useState(false);
	const [error, setError] = useState<string | null>(null);

	const canTest = selectedDataset && selectedModel;

	async function handleTestModel() {
		if (!canTest || loading) return;

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
						images: result.data.images,
						predictionErrors: result.data.prediction_errors ?? []
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
		<div className="section-top">
			{/* Identity cards */}
			<div className="grid-2col-mb">
				<div className="info-card">
					<p className="info-card-label">Selected Dataset</p>
					<p className="info-card-value">{selectedDataset || <em className="empty-state">None</em>}</p>
				</div>
				<div className="info-card">
					<p className="info-card-label">Selected Model</p>
					<p className="info-card-value">
						{selectedModel ? (
							`${capitalize(selectedModel.model_name)} ${MODEL_TYPE_SHORT_NAMES[selectedModel.model_type] ?? selectedModel.model_type}`
						) : (
							<em className="empty-state">None</em>
						)}
					</p>
				</div>
			</div>

			<Button variant="contained" onClick={handleTestModel} disabled={!canTest}>
				Test Model on Dataset
			</Button>

			<LoadingProgress isLoading={loading} message="Applying a model to your data..." />

			{!canTest && (
				<Alert severity="warning" sx={{ mt: 2 }}>
					Please select both a dataset and a model to test.
				</Alert>
			)}

			{error && (
				<div role="alert" className="error-banner">
					<strong>Error:</strong> {error}
				</div>
			)}
		</div>
	);
}
