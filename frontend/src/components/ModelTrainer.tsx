import { Button } from '@mui/material';
import { useTheme } from '@mui/material/styles';
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

export default function ModelTrainer({ apiBase, xApiKey, selectedDataset, selectedModel, onTestComplete }: ModelTrainerProps) {
	const theme = useTheme();
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
		<div style={{ marginTop: '2rem' }}>
			{/* Identity cards */}
			<div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1.5rem' }}>
				<div style={{ padding: '1rem', borderRadius: '6px', border: `1px solid ${theme.palette.divider}` }}>
					<p style={{ margin: 0, fontSize: '0.75rem', color: theme.palette.text.secondary, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
						Selected Dataset
					</p>
					<p style={{ margin: '0.4rem 0 0 0', fontSize: '1.1rem', fontWeight: 'bold' }}>
						{selectedDataset || <em style={{ opacity: 0.6, fontStyle: 'italic' }}>None</em>}
					</p>
				</div>
				<div style={{ padding: '1rem', borderRadius: '6px', border: `1px solid ${theme.palette.divider}` }}>
					<p style={{ margin: 0, fontSize: '0.75rem', color: theme.palette.text.secondary, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
						Selected Model
					</p>
					<p style={{ margin: '0.4rem 0 0 0', fontSize: '1.1rem', fontWeight: 'bold' }}>
						{selectedModel
							? `${capitalize(selectedModel.model_name)} ${MODEL_TYPE_SHORT_NAMES[selectedModel.model_type] ?? selectedModel.model_type}`
							: <em style={{ opacity: 0.6, fontStyle: 'italic' }}>None</em>}
					</p>
				</div>
			</div>

			<Button
				variant="contained"
				onClick={handleTestModel}
				disabled={!canTest || loading}
				sx={{ backgroundColor: '#452ee4', '&:hover': { backgroundColor: '#241291' } }}
			>
				Test Model on Dataset
			</Button>

			<LoadingProgress isLoading={loading} message="Applying a model to your data..." />

			{!canTest && <p style={{ marginTop: '1rem', color: '#ff9800' }}>Please select both a dataset and a model to test.</p>}

			{error && (
				<div
					style={{
						marginTop: '1rem',
						padding: '1rem',
						backgroundColor: theme.palette.mode === 'dark' ? '#3d1a1a' : 'rgba(244, 67, 54, 0.08)',
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
