import { ThemeProvider, createTheme } from '@mui/material/styles';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import DatasetSelector from './DatasetSelector.js';

const theme = createTheme();

function renderWithTheme(ui: React.ReactElement) {
	return render(<ThemeProvider theme={theme}>{ui}</ThemeProvider>);
}

describe('DatasetSelector', () => {
	it('shows fallback message when datasets list is empty', () => {
		renderWithTheme(<DatasetSelector datasets={[]} selected="" onSelect={vi.fn()} />);
		expect(screen.getByText(/Failed to load datasets/i)).toBeInTheDocument();
	});

	it('renders a select field when datasets are provided', () => {
		renderWithTheme(<DatasetSelector datasets={['alpha', 'beta']} selected="" onSelect={vi.fn()} />);
		expect(screen.getByRole('combobox')).toBeInTheDocument();
	});

	it('reflects the selected value', () => {
		renderWithTheme(<DatasetSelector datasets={['alpha', 'beta']} selected="alpha" onSelect={vi.fn()} />);
		expect(screen.getByRole('combobox')).toHaveTextContent('alpha');
	});

	it('is disabled when disabled prop is true', () => {
		renderWithTheme(<DatasetSelector datasets={['alpha']} selected="" onSelect={vi.fn()} disabled />);
		expect(screen.getByRole('combobox')).toHaveAttribute('aria-disabled', 'true');
	});

	it('calls onSelect when a dataset is chosen', async () => {
		const onSelect = vi.fn();
		renderWithTheme(<DatasetSelector datasets={['alpha', 'beta']} selected="" onSelect={onSelect} />);
		const user = userEvent.setup();
		await user.click(screen.getByRole('combobox'));
		await user.click(screen.getByRole('option', { name: 'beta' }));
		expect(onSelect).toHaveBeenCalledWith('beta');
	});

	it('renders datasets sorted alphabetically', async () => {
		renderWithTheme(<DatasetSelector datasets={['zeta', 'alpha', 'mango']} selected="" onSelect={vi.fn()} />);
		const user = userEvent.setup();
		await user.click(screen.getByRole('combobox'));
		const options = screen.getAllByRole('option').filter((o) => o.getAttribute('data-value') !== '');
		const labels = options.map((o) => o.textContent);
		expect(labels).toEqual([...labels].sort());
	});
});
