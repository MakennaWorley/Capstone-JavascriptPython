import { ThemeProvider, createTheme } from '@mui/material/styles';
import { render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import DatasetDashboard from './DatasetDisplayDashboard.js';

const theme = createTheme();

function renderWithTheme(ui: React.ReactElement) {
	return render(<ThemeProvider theme={theme}>{ui}</ThemeProvider>);
}

const defaultProps = {
	apiBase: 'http://localhost:8000',
	xApiKey: 'test-key',
	selectedDataset: ''
};

const csvHeaders = 'site,i_0000,i_0001\n';
const observedCsv = `${csvHeaders}S1,0,1\nS2,1,0\n`;
const truthCsv = `${csvHeaders}S1,0,1\nS2,1,0\n`;

describe('DatasetDashboard', () => {
	beforeEach(() => {
		vi.stubGlobal('fetch', vi.fn());
	});

	afterEach(() => {
		vi.restoreAllMocks();
	});

	it('renders without crashing when no dataset is selected', () => {
		renderWithTheme(<DatasetDashboard {...defaultProps} />);
		// Should render an empty container without errors
	});

	it('fetches dashboard data when a dataset is selected', async () => {
		vi.mocked(fetch).mockResolvedValueOnce({
			ok: true,
			headers: { get: () => 'application/json' },
			json: async () => ({
				status: 'success',
				data: {
					observed_genotypes_csv: observedCsv,
					truth_genotypes_csv: truthCsv
				}
			})
		} as unknown as Response);

		renderWithTheme(<DatasetDashboard {...defaultProps} selectedDataset="my_dataset" />);

		await waitFor(() => {
			expect(fetch).toHaveBeenCalledWith(
				expect.stringContaining('/api/dataset/my_dataset/dashboard'),
				expect.objectContaining({ headers: expect.objectContaining({ 'x-api-key': 'test-key' }) })
			);
		});
	});

	it('renders CSV preview table when data is loaded', async () => {
		vi.mocked(fetch).mockResolvedValueOnce({
			ok: true,
			headers: { get: () => 'application/json' },
			json: async () => ({
				status: 'success',
				data: {
					observed_genotypes_csv: observedCsv,
					truth_genotypes_csv: truthCsv
				}
			})
		} as unknown as Response);

		renderWithTheme(<DatasetDashboard {...defaultProps} selectedDataset="my_dataset" />);

		await waitFor(() => {
			expect(screen.getByText(/Genotypes \(preview\)/i)).toBeInTheDocument();
		});
	});

	it('re-fetches when selectedDataset changes', async () => {
		vi.mocked(fetch).mockResolvedValue({
			ok: true,
			headers: { get: () => 'application/json' },
			json: async () => ({
				status: 'success',
				data: { observed_genotypes_csv: observedCsv, truth_genotypes_csv: truthCsv }
			})
		} as unknown as Response);

		const { rerender } = renderWithTheme(<DatasetDashboard {...defaultProps} selectedDataset="dataset_1" />);

		await waitFor(() => expect(fetch).toHaveBeenCalledTimes(1));

		rerender(
			<ThemeProvider theme={theme}>
				<DatasetDashboard {...defaultProps} selectedDataset="dataset_2" />
			</ThemeProvider>
		);

		await waitFor(() => expect(fetch).toHaveBeenCalledTimes(2));
	});

	it('does not crash when the API returns an error response', async () => {
		vi.mocked(fetch).mockResolvedValueOnce({
			ok: false,
			headers: { get: () => 'application/json' },
			json: async () => ({ status: 'error', message: 'not found' })
		} as unknown as Response);

		renderWithTheme(<DatasetDashboard {...defaultProps} selectedDataset="bad_dataset" />);
		// Should not throw — component silently handles failures
		await waitFor(() => expect(fetch).toHaveBeenCalledTimes(1));
	});
});
