import { ThemeProvider, createTheme } from '@mui/material/styles';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import ModelTrainer from './ModelTrainer.js';

const theme = createTheme();

function renderWithTheme(ui: React.ReactElement) {
	return render(<ThemeProvider theme={theme}>{ui}</ThemeProvider>);
}

const defaultProps = {
	apiBase: 'http://localhost:8000',
	xApiKey: 'test-key',
	selectedDataset: '',
	selectedModel: null
};

const filledProps = {
	apiBase: 'http://localhost:8000',
	xApiKey: 'test-key',
	selectedDataset: 'my_dataset',
	selectedModel: { model_name: 'small', model_type: 'dnn_dosage' }
};

describe('ModelTrainer', () => {
	beforeEach(() => {
		vi.stubGlobal('fetch', vi.fn());
	});

	afterEach(() => {
		vi.restoreAllMocks();
	});

	it('renders the Test Model button', () => {
		renderWithTheme(<ModelTrainer {...defaultProps} />);
		expect(screen.getByRole('button', { name: /Test Model on Dataset/i })).toBeInTheDocument();
	});

	it('button is disabled when no dataset and model are selected', () => {
		renderWithTheme(<ModelTrainer {...defaultProps} />);
		expect(screen.getByRole('button', { name: /Test Model on Dataset/i })).toBeDisabled();
	});

	it('shows a warning when no dataset or model is selected', () => {
		renderWithTheme(<ModelTrainer {...defaultProps} />);
		expect(screen.getByRole('alert')).toHaveTextContent(/select both a dataset and a model/i);
	});

	it('button is enabled when both dataset and model are selected', () => {
		renderWithTheme(<ModelTrainer {...filledProps} />);
		expect(screen.getByRole('button', { name: /Test Model on Dataset/i })).toBeEnabled();
	});

	it('displays the selected dataset name', () => {
		renderWithTheme(<ModelTrainer {...filledProps} />);
		expect(screen.getByText('my_dataset')).toBeInTheDocument();
	});

	it('displays the selected model name', () => {
		renderWithTheme(<ModelTrainer {...filledProps} />);
		expect(screen.getByText('Small Deep Neural Network')).toBeInTheDocument();
	});

	it('calls onTestComplete with response data on successful fetch', async () => {
		const onTestComplete = vi.fn();
		const apiResponse = {
			status: 'success',
			message: 'OK',
			data: {
				test_metrics: { accuracy: 0.9 },
				paths: { graph_test: 'a', graph_cm: 'b', model_dir: 'c' },
				images: {},
				prediction_errors: []
			}
		};
		vi.mocked(fetch).mockResolvedValueOnce({
			json: async () => apiResponse
		} as Response);

		renderWithTheme(<ModelTrainer {...filledProps} onTestComplete={onTestComplete} />);
		const user = userEvent.setup();
		await user.click(screen.getByRole('button', { name: /Test Model on Dataset/i }));

		await waitFor(() => {
			expect(onTestComplete).toHaveBeenCalledWith(expect.objectContaining({ testMetrics: { accuracy: 0.9 } }));
		});
	});

	it('shows an error alert when fetch returns an error status', async () => {
		vi.mocked(fetch).mockResolvedValueOnce({
			json: async () => ({ status: 'error', message: 'Something went wrong' })
		} as Response);

		renderWithTheme(<ModelTrainer {...filledProps} />);
		const user = userEvent.setup();
		await user.click(screen.getByRole('button', { name: /Test Model on Dataset/i }));

		await waitFor(() => {
			expect(screen.getByRole('alert')).toHaveTextContent(/Something went wrong/i);
		});
	});

	it('shows a network error when fetch throws', async () => {
		vi.mocked(fetch).mockRejectedValueOnce(new Error('Network failure'));

		renderWithTheme(<ModelTrainer {...filledProps} />);
		const user = userEvent.setup();
		await user.click(screen.getByRole('button', { name: /Test Model on Dataset/i }));

		await waitFor(() => {
			expect(screen.getByRole('alert')).toHaveTextContent(/Network failure/i);
		});
	});
});
