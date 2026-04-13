import { ThemeProvider, createTheme } from '@mui/material/styles';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import DatasetModelCreationForm from './DatasetModelCreationForm.js';

const theme = createTheme();

function renderWithTheme(ui: React.ReactElement) {
	return render(<ThemeProvider theme={theme}>{ui}</ThemeProvider>);
}

const defaultProps = {
	apiBase: 'http://localhost:8000',
	xApiKey: 'test-key'
};

describe('DatasetModelCreationForm', () => {
	beforeEach(() => {
		vi.stubGlobal('fetch', vi.fn());
	});

	afterEach(() => {
		vi.restoreAllMocks();
	});

	it('renders the basic settings section', () => {
		renderWithTheme(<DatasetModelCreationForm {...defaultProps} />);
		expect(screen.getByText(/Basic Settings/i)).toBeInTheDocument();
	});

	it('renders the Dataset name field', () => {
		renderWithTheme(<DatasetModelCreationForm {...defaultProps} />);
		expect(screen.getByLabelText(/Dataset name/i)).toBeInTheDocument();
	});

	it('renders the Advanced Settings checkbox', () => {
		renderWithTheme(<DatasetModelCreationForm {...defaultProps} />);
		expect(screen.getByRole('checkbox', { name: /Advanced Settings/i })).toBeInTheDocument();
	});

	it('shows Advanced Settings inputs when checkbox is checked', async () => {
		renderWithTheme(<DatasetModelCreationForm {...defaultProps} />);
		const user = userEvent.setup();
		await user.click(screen.getByRole('checkbox', { name: /Advanced Settings/i }));
		expect(screen.getByLabelText(/Sequence length/i)).toBeInTheDocument();
	});

	it('shows a validation error when name is empty and submit attempted', async () => {
		renderWithTheme(<DatasetModelCreationForm {...defaultProps} />);
		const user = userEvent.setup();
		await user.click(screen.getByRole('button', { name: /Generate Dataset/i }));
		expect(screen.getByText(/Dataset name is required/i)).toBeInTheDocument();
	});

	it('shows a validation error when name has spaces or special chars', async () => {
		renderWithTheme(<DatasetModelCreationForm {...defaultProps} />);
		const user = userEvent.setup();
		await user.type(screen.getByLabelText(/Dataset name/i), 'bad name!');
		await user.click(screen.getByRole('button', { name: /Generate Dataset/i }));
		expect(screen.getByText(/alphanumeric only/i)).toBeInTheDocument();
	});

	it('calls fetch on valid submission', async () => {
		vi.mocked(fetch).mockResolvedValueOnce({
			ok: true,
			text: async () => JSON.stringify({ status: 'success' }),
			status: 200
		} as unknown as Response);

		const onSuccess = vi.fn();
		renderWithTheme(<DatasetModelCreationForm {...defaultProps} onSuccess={onSuccess} />);
		const user = userEvent.setup();
		await user.type(screen.getByLabelText(/Dataset name/i), 'validname01');
		await user.click(screen.getByRole('button', { name: /Generate Dataset/i }));

		await waitFor(() => {
			expect(fetch).toHaveBeenCalledWith(
				expect.stringContaining('/api/create/data'),
				expect.objectContaining({ method: 'POST' })
			);
		});
	});

	it('calls onSuccess callback after successful submission', async () => {
		vi.mocked(fetch).mockResolvedValueOnce({
			ok: true,
			text: async () => JSON.stringify({ status: 'success' }),
			status: 200
		} as unknown as Response);

		const onSuccess = vi.fn();
		renderWithTheme(<DatasetModelCreationForm {...defaultProps} onSuccess={onSuccess} />);
		const user = userEvent.setup();
		await user.type(screen.getByLabelText(/Dataset name/i), 'validname01');
		await user.click(screen.getByRole('button', { name: /Generate Dataset/i }));

		await waitFor(() => {
			expect(onSuccess).toHaveBeenCalledTimes(1);
		});
	});

	it('shows an error message when the API returns a non-ok response', async () => {
		vi.mocked(fetch).mockResolvedValueOnce({
			ok: false,
			status: 500,
			text: async () => JSON.stringify({ message: 'Internal server error' })
		} as unknown as Response);

		renderWithTheme(<DatasetModelCreationForm {...defaultProps} />);
		const user = userEvent.setup();
		await user.type(screen.getByLabelText(/Dataset name/i), 'validname01');
		await user.click(screen.getByRole('button', { name: /Generate Dataset/i }));

		await waitFor(() => {
			expect(screen.getByText(/Internal server error/i)).toBeInTheDocument();
		});
	});

	it('displays the network error message when fetch throws', async () => {
		vi.mocked(fetch).mockRejectedValueOnce(new Error('Network fail'));

		renderWithTheme(<DatasetModelCreationForm {...defaultProps} />);
		const user = userEvent.setup();
		await user.type(screen.getByLabelText(/Dataset name/i), 'validname01');
		await user.click(screen.getByRole('button', { name: /Generate Dataset/i }));

		await waitFor(() => {
			expect(screen.getByText(/Network error/i)).toBeInTheDocument();
		});
	});

	it('shows advanced validation errors when advanced config is invalid', async () => {
		renderWithTheme(<DatasetModelCreationForm {...defaultProps} />);
		const user = userEvent.setup();
		await user.type(screen.getByLabelText(/Dataset name/i), 'validname01');
		await user.click(screen.getByRole('checkbox', { name: /Advanced Settings/i }));

		// Clear sequence length and enter 0
		const seqField = screen.getByLabelText(/Sequence length/i);
		await user.clear(seqField);
		await user.type(seqField, '0');
		await user.click(screen.getByRole('button', { name: /Generate Dataset/i }));

		expect(screen.getByText(/Sequence length must be a positive number/i)).toBeInTheDocument();
	});
});
