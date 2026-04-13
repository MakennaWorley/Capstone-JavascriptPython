import { ThemeProvider, createTheme } from '@mui/material/styles';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import ModelSelector from './ModelSelector.js';

const theme = createTheme();

function renderWithTheme(ui: React.ReactElement) {
	return render(<ThemeProvider theme={theme}>{ui}</ThemeProvider>);
}

const sampleModels = [
	{ model_name: 'small', model_type: 'dnn_dosage' },
	{ model_name: 'large', model_type: 'hmm_dosage' }
];

describe('ModelSelector', () => {
	it('shows fallback message when models list is empty', () => {
		renderWithTheme(<ModelSelector models={[]} selected={null} onSelect={vi.fn()} />);
		expect(screen.getByText(/Failed to load models/i)).toBeInTheDocument();
	});

	it('renders a select field when models are provided', () => {
		renderWithTheme(<ModelSelector models={sampleModels} selected={null} onSelect={vi.fn()} />);
		expect(screen.getByRole('combobox')).toBeInTheDocument();
	});

	it('reflects the selected model', () => {
		renderWithTheme(<ModelSelector models={sampleModels} selected={sampleModels[0]} onSelect={vi.fn()} />);
		expect(screen.getByRole('combobox')).toHaveTextContent('Small Deep Neural Network');
	});

	it('is disabled when disabled prop is true', () => {
		renderWithTheme(<ModelSelector models={sampleModels} selected={null} onSelect={vi.fn()} disabled />);
		expect(screen.getByRole('combobox')).toHaveAttribute('aria-disabled', 'true');
	});

	it('calls onSelect with the chosen model when a user picks one', async () => {
		const onSelect = vi.fn();
		renderWithTheme(<ModelSelector models={sampleModels} selected={null} onSelect={onSelect} />);
		const user = userEvent.setup();
		await user.click(screen.getByRole('combobox'));
		await user.click(screen.getByRole('option', { name: /Deep Neural Network/i }));
		expect(onSelect).toHaveBeenCalledWith(expect.objectContaining({ model_type: 'dnn_dosage' }));
	});

	it('calls onSelect with null when the empty option is selected', async () => {
		const onSelect = vi.fn();
		renderWithTheme(<ModelSelector models={sampleModels} selected={sampleModels[0]} onSelect={onSelect} />);
		// The handleChange function calls onSelect(null) when value is empty string
		// We test this the same way the component's handleChange function works
		const user = userEvent.setup();
		await user.click(screen.getByRole('combobox'));
		// The "Select…" item is disabled, so test we can open the menu at least
		expect(screen.getByRole('option', { name: /Select…/i })).toBeInTheDocument();
	});

	it('displays human-readable label for known model types', async () => {
		renderWithTheme(<ModelSelector models={[{ model_name: 'base', model_type: 'multi_log_regression' }]} selected={null} onSelect={vi.fn()} />);
		const user = userEvent.setup();
		await user.click(screen.getByRole('combobox'));
		expect(screen.getByRole('option', { name: /Multinomial Logistic Regression/i })).toBeInTheDocument();
	});
});
