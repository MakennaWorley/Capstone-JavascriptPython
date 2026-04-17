import { renderHook, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { useDatasetsPoll, useModelsPoll } from './useApiPolling.js';

beforeEach(() => {
	vi.stubGlobal('fetch', vi.fn());
});

afterEach(() => {
	vi.restoreAllMocks();
});

describe('useDatasetsPoll', () => {
	it('starts with an empty datasets array and isLoading true', () => {
		vi.mocked(fetch).mockReturnValue(new Promise(() => {}));
		const { result } = renderHook(() => useDatasetsPoll('http://localhost:8000', 'test-key', 5000));
		expect(result.current.datasets).toEqual([]);
		expect(result.current.isLoading).toBe(true);
	});

	it('populates datasets after a successful fetch', async () => {
		vi.mocked(fetch).mockResolvedValue({
			json: async () => ({
				status: 'success',
				message: 'OK',
				data: { datasets: ['alpha', 'beta'], count: 2 }
			})
		} as Response);

		const { result } = renderHook(() => useDatasetsPoll('http://localhost:8000', 'test-key', 5000));

		await waitFor(() => {
			expect(result.current.datasets).toEqual(['alpha', 'beta']);
		});

		expect(result.current.isLoading).toBe(false);
		expect(result.current.error).toBeNull();
	});

	it('sets error when fetch fails', async () => {
		vi.mocked(fetch).mockRejectedValue(new Error('Network failure'));

		const { result } = renderHook(() => useDatasetsPoll('http://localhost:8000', 'test-key', 5000));

		await waitFor(() => {
			expect(result.current.error).toBeTruthy();
		});

		expect(result.current.isLoading).toBe(false);
	});

	it('refresh increments the tick and re-subscribes', async () => {
		vi.mocked(fetch).mockResolvedValue({
			json: async () => ({
				status: 'success',
				message: 'OK',
				data: { datasets: ['alpha'], count: 1 }
			})
		} as Response);

		const { result } = renderHook(() => useDatasetsPoll('http://localhost:8000', 'test-key', 5000));

		await waitFor(() => expect(result.current.datasets).toEqual(['alpha']));

		result.current.refresh();

		// fetch should be called at least twice (initial + refresh)
		await waitFor(() => expect(fetch).toHaveBeenCalledTimes(2));
	});
});

describe('useModelsPoll', () => {
	it('starts with an empty models array', () => {
		vi.mocked(fetch).mockReturnValue(new Promise(() => {}));
		const { result } = renderHook(() => useModelsPoll('http://localhost:8000', 'test-key', 5000));
		expect(result.current.models).toEqual([]);
		expect(result.current.isLoading).toBe(true);
	});

	it('populates models after a successful fetch', async () => {
		const models = [{ model_name: 'small', model_type: 'dnn_dosage' }];
		vi.mocked(fetch).mockResolvedValue({
			json: async () => ({
				status: 'success',
				message: 'OK',
				data: { models, count: 1 }
			})
		} as Response);

		const { result } = renderHook(() => useModelsPoll('http://localhost:8000', 'test-key', 5000));

		await waitFor(() => {
			expect(result.current.models).toEqual(models);
		});

		expect(result.current.isLoading).toBe(false);
		expect(result.current.error).toBeNull();
	});

	it('sets error when models fetch fails', async () => {
		vi.mocked(fetch).mockRejectedValue(new Error('Network failure'));

		const { result } = renderHook(() => useModelsPoll('http://localhost:8000', 'test-key', 5000));

		await waitFor(() => {
			expect(result.current.error).toBeTruthy();
		});
	});
});
