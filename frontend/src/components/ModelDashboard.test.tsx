import { ThemeProvider, createTheme } from '@mui/material/styles';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';
import ModelDashboard from './ModelDashboard.js';

const theme = createTheme();

function renderWithTheme(ui: React.ReactElement) {
	return render(<ThemeProvider theme={theme}>{ui}</ThemeProvider>);
}

describe('ModelDashboard', () => {
	it('renders the Model Dashboard heading', () => {
		renderWithTheme(<ModelDashboard model={{ model_name: 'small', model_type: 'dnn_dosage' }} />);
		expect(screen.getByRole('heading', { name: /Model Dashboard/i })).toBeInTheDocument();
	});

	it('shows the model type label for a known type', () => {
		renderWithTheme(<ModelDashboard model={{ model_name: 'small', model_type: 'dnn_dosage' }} />);
		expect(screen.getByText('Deep Neural Network')).toBeInTheDocument();
	});

	it('shows the model size / name', () => {
		renderWithTheme(<ModelDashboard model={{ model_name: 'small', model_type: 'dnn_dosage' }} />);
		expect(screen.getByText('Small')).toBeInTheDocument();
	});

	it('shows Custom Model label for an unknown model type', () => {
		renderWithTheme(<ModelDashboard model={{ model_name: 'custom', model_type: 'unknown_type' }} />);
		expect(screen.getByText('Custom Model')).toBeInTheDocument();
	});

	it('renders the Stats for Nerds toggle', () => {
		renderWithTheme(<ModelDashboard model={{ model_name: 'small', model_type: 'dnn_dosage' }} />);
		expect(screen.getByLabelText(/Stats for Nerds/i)).toBeInTheDocument();
	});

	it('switches to technical description when Stats for Nerds is toggled', async () => {
		renderWithTheme(<ModelDashboard model={{ model_name: 'small', model_type: 'dnn_dosage' }} />);
		const user = userEvent.setup();
		const toggle = screen.getByLabelText(/Stats for Nerds/i);
		// Simple (non-nerd) description should mention "pattern-recognition"
		expect(screen.getByText(/pattern-recognition/i)).toBeInTheDocument();
		await user.click(toggle);
		// Nerd description should mention "batch normalization" or similar technical terms
		expect(screen.getByText(/batch normalization/i)).toBeInTheDocument();
	});

	it('renders all five model types without crashing', () => {
		const types = ['dnn_dosage', 'multi_log_regression', 'hmm_dosage', 'gnn_dosage', 'bayes_softmax3'];
		for (const model_type of types) {
			const { unmount } = renderWithTheme(<ModelDashboard model={{ model_name: 'small', model_type }} />);
			expect(screen.getByRole('heading', { name: /Model Dashboard/i })).toBeInTheDocument();
			unmount();
		}
	});
});
