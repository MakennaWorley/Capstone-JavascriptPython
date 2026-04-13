import { ThemeProvider, createTheme } from '@mui/material/styles';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';
import ModelStats from './ModelStats.js';

const theme = createTheme();

function renderWithTheme(ui: React.ReactElement) {
	return render(<ThemeProvider theme={theme}>{ui}</ThemeProvider>);
}

const sampleMetrics = {
	model: 'small',
	dataset: 'test_dataset',
	accuracy: 0.85,
	f1_macro: 0.72,
	auc_macro: 0.91
};

const samplePaths = {
	graph_test: '/path/to/graph',
	graph_cm: '/path/to/cm',
	model_dir: '/path/to/model'
};

describe('ModelStats', () => {
	it('shows placeholder message when no data is provided', () => {
		renderWithTheme(<ModelStats paths={null} testMetrics={null} images={null} />);
		expect(screen.getByText(/No test results available/i)).toBeInTheDocument();
	});

	it('renders the heading', () => {
		renderWithTheme(<ModelStats paths={samplePaths} testMetrics={sampleMetrics} images={null} />);
		expect(screen.getByRole('heading', { name: /Model Statistics/i })).toBeInTheDocument();
	});

	it('displays model and dataset names in the summary sentence', () => {
		renderWithTheme(<ModelStats paths={samplePaths} testMetrics={sampleMetrics} images={null} />);
		expect(screen.getByText(/small/)).toBeInTheDocument();
		expect(screen.getByText(/test_dataset/)).toBeInTheDocument();
	});

	it('renders metric rows in the table', () => {
		renderWithTheme(<ModelStats paths={samplePaths} testMetrics={sampleMetrics} images={null} />);
		expect(screen.getByText('Accuracy')).toBeInTheDocument();
		expect(screen.getByText('85.00%')).toBeInTheDocument();
	});

	it('toggles to nerd descriptions when Stats for Nerds switch is clicked', async () => {
		renderWithTheme(<ModelStats paths={samplePaths} testMetrics={sampleMetrics} images={null} />);
		const toggle = screen.getByLabelText(/Stats for Nerds/i);
		expect(toggle).not.toBeChecked();
		const user = userEvent.setup();
		await user.click(toggle);
		expect(toggle).toBeChecked();
	});

	it('shows prediction error count when predictionErrors are provided', () => {
		const errors = [
			{ individual: 'ind1', site: 'S1', predicted: 1, actual: 0 },
			{ individual: 'ind2', site: 'S1', predicted: 2, actual: 1 }
		];
		renderWithTheme(<ModelStats paths={samplePaths} testMetrics={sampleMetrics} images={null} predictionErrors={errors} />);
		expect(screen.getByText(/2 errors across 1 site/i)).toBeInTheDocument();
	});

	it('shows raw JSON block when debugMode is true', () => {
		renderWithTheme(<ModelStats paths={samplePaths} testMetrics={sampleMetrics} images={null} debugMode={true} />);
		expect(screen.getByText(/Raw Response/i)).toBeInTheDocument();
	});

	it('does not show raw JSON block when debugMode is false', () => {
		renderWithTheme(<ModelStats paths={samplePaths} testMetrics={sampleMetrics} images={null} debugMode={false} />);
		expect(screen.queryByText(/Raw Response/i)).not.toBeInTheDocument();
	});
});
