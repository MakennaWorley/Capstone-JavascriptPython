import { useState } from 'react';

type Model = {
	model_name: string;
	model_type: string;
};

type ModelTrainerProps = {
	apiBase: string;
	xApiKey: string;
	selectedDataset: string;
	selectedModel: Model | null;
};

type ApiSuccessTest = {
	status: 'success';
	message: string;
	data: {
		log: string;
		test_metrics: any;
		paths: {
			graph_test: string;
			graph_cm: string;
			model_dir: string;
		};
	};
};

type ApiError = {
	status: 'error';
	code?: string;
	message: string;
};

export default function ModelTrainer({ apiBase, xApiKey, selectedDataset, selectedModel }: ModelTrainerProps) {
	const [loading, setLoading] = useState(false);
	const [testResult, setTestResult] = useState<string | null>(null);
	const [error, setError] = useState<string | null>(null);

	const canTest = selectedDataset && selectedModel;

	async function handleTestModel() {
		if (!canTest) return;

		setLoading(true);
		setError(null);
		setTestResult(null);

		try {
			const response = await fetch(`${apiBase}/api/models/test`, {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json',
					'x-api-key': xApiKey,
				},
				body: JSON.stringify({
					dataset_name: selectedDataset,
					model_name: selectedModel.model_name,
					model_type: selectedModel.model_type,
				}),
			});

			const result: ApiSuccessTest | ApiError = await response.json();

			if (result.status === 'success') {
				setTestResult(result.data.log);
				setError(null);
			} else {
				setError(result.message || 'Failed to test model');
				setTestResult(null);
			}
		} catch (err) {
			setError(`Error testing model: ${err}`);
			setTestResult(null);
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
				onClick={handleTestModel}
				disabled={!canTest || loading}
				style={{
					padding: '0.6rem 1.2rem',
					backgroundColor: canTest && !loading ? '#4CAF50' : '#555',
					color: 'white',
					border: 'none',
					borderRadius: '4px',
					cursor: canTest && !loading ? 'pointer' : 'not-allowed',
					fontSize: '1rem',
				}}
			>
				{loading ? 'Testing...' : 'Test Model on Dataset'}
			</button>

			{!canTest && (
				<p style={{ marginTop: '1rem', color: '#ff9800' }}>
					Please select both a dataset and a model to test.
				</p>
			)}

			{error && (
				<div
					style={{
						marginTop: '1rem',
						padding: '1rem',
						backgroundColor: '#3d1a1a',
						borderLeft: '4px solid #f44336',
						borderRadius: '4px',
					}}
				>
					<strong>Error:</strong> {error}
				</div>
			)}

			{testResult && (
				<div style={{ marginTop: '1.5rem' }}>
					<h4>Test Results</h4>
					<pre
						style={{
							backgroundColor: '#0a0a0a',
							padding: '1rem',
							borderRadius: '4px',
							overflowX: 'auto',
							fontSize: '0.85rem',
							lineHeight: '1.5',
							border: '1px solid #333',
							maxHeight: '500px',
							overflowY: 'auto',
						}}
					>
						{testResult}
					</pre>
				</div>
			)}
		</div>
	);
}
