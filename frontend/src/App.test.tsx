import { render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import App from './App.js';

beforeEach(() => {
	vi.stubGlobal('fetch', vi.fn());
	// Provide stable env vars
	vi.stubGlobal('import.meta', {
		env: { VITE_API_BASE: 'http://localhost:8000', VITE_X_API_KEY: 'test-key' }
	});

	// Return empty lists by default so polling doesn't crash the tests
	vi.mocked(fetch).mockResolvedValue({
		json: async () => ({ status: 'success', message: 'OK', data: { datasets: [], count: 0, models: [], count2: 0 } })
	} as unknown as Response);
});

afterEach(() => {
	vi.restoreAllMocks();
});

describe('App', () => {
	it('renders without crashing', async () => {
		render(<App />);
		await waitFor(() => expect(document.body).toBeTruthy());
	});

	it('renders the main content region', async () => {
		render(<App />);
		await waitFor(() => {
			expect(screen.getByRole('main')).toBeInTheDocument();
		});
	});

	it('renders the Skip to main content accessibility link', async () => {
		render(<App />);
		await waitFor(() => {
			expect(screen.getByText(/Skip to main content/i)).toBeInTheDocument();
		});
	});

	it('renders the Selection heading', async () => {
		render(<App />);
		await waitFor(() => {
			expect(screen.getByRole('heading', { name: /Selection/i })).toBeInTheDocument();
		});
	});

	it('renders the navigation sidebar', async () => {
		render(<App />);
		await waitFor(() => {
			expect(screen.getByRole('navigation', { name: /Main navigation/i })).toBeInTheDocument();
		});
	});

	it('renders the dataset selector combobox when datasets load', async () => {
		vi.mocked(fetch).mockResolvedValue({
			json: async () => ({
				status: 'success',
				message: 'OK',
				data: { datasets: ['alpha', 'beta'], count: 2, models: [] }
			})
		} as Response);

		render(<App />);
		// The DatasetSelector shows a fallback message when datasets are empty, so we wait for data
		await waitFor(() => {
			expect(screen.getAllByRole('combobox').length).toBeGreaterThan(0);
		});
	});
});
