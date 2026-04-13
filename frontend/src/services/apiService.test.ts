import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import ApiService from './apiService.js';

describe('ApiService', () => {
	beforeEach(() => {
		vi.stubGlobal('fetch', vi.fn());
		vi.useFakeTimers();
	});

	afterEach(() => {
		vi.restoreAllMocks();
		vi.useRealTimers();
	});

	describe('createDatasetsPoll', () => {
		it('emits an array of dataset strings on success', async () => {
			vi.mocked(fetch).mockResolvedValue({
				json: async () => ({
					status: 'success',
					message: 'OK',
					data: { datasets: ['alpha', 'beta'], count: 2 }
				})
			} as Response);

			const service = new ApiService('http://localhost:8000', 'test-key');
			const result = await new Promise<string[]>((resolve, reject) => {
				service.createDatasetsPoll(100).subscribe({ next: resolve, error: reject });
				vi.advanceTimersByTime(0);
			});

			expect(result).toEqual(['alpha', 'beta']);
		});

		it('sends the x-api-key header', async () => {
			vi.mocked(fetch).mockResolvedValue({
				json: async () => ({
					status: 'success',
					message: 'OK',
					data: { datasets: [], count: 0 }
				})
			} as Response);

			const service = new ApiService('http://localhost:8000', 'my-secret');
			await new Promise<string[]>((resolve, reject) => {
				service.createDatasetsPoll(100).subscribe({ next: resolve, error: reject });
				vi.advanceTimersByTime(0);
			});

			expect(fetch).toHaveBeenCalledWith(
				expect.stringContaining('/api/datasets/list'),
				expect.objectContaining({ headers: expect.objectContaining({ 'x-api-key': 'my-secret' }) })
			);
		});

		it('emits an error when the API returns an error status', async () => {
			vi.mocked(fetch).mockResolvedValue({
				json: async () => ({ status: 'error', message: 'Unauthorized' })
			} as Response);

			const service = new ApiService('http://localhost:8000', 'test-key');
			const error = await new Promise<Error>((resolve) => {
				service.createDatasetsPoll(100).subscribe({ next: () => {}, error: resolve });
				vi.advanceTimersByTime(0);
			});

			expect(error.message).toBe('Unauthorized');
		});

		it('emits an error when fetch rejects', async () => {
			vi.mocked(fetch).mockRejectedValue(new Error('Network failure'));

			const service = new ApiService('http://localhost:8000', 'test-key');
			const error = await new Promise<Error>((resolve) => {
				service.createDatasetsPoll(100).subscribe({ next: () => {}, error: resolve });
				vi.advanceTimersByTime(0);
			});

			expect(error.message).toBe('Network failure');
		});
	});

	describe('createModelsPoll', () => {
		it('emits an array of model objects on success', async () => {
			const models = [{ model_name: 'small', model_type: 'dnn_dosage' }];
			vi.mocked(fetch).mockResolvedValue({
				json: async () => ({
					status: 'success',
					message: 'OK',
					data: { models, count: 1 }
				})
			} as Response);

			const service = new ApiService('http://localhost:8000', 'test-key');
			const result = await new Promise<typeof models>((resolve, reject) => {
				service.createModelsPoll(100).subscribe({ next: resolve, error: reject });
				vi.advanceTimersByTime(0);
			});

			expect(result).toEqual(models);
		});

		it('emits an error when models fetch rejects', async () => {
			vi.mocked(fetch).mockRejectedValue(new Error('Network failure'));

			const service = new ApiService('http://localhost:8000', 'test-key');
			const error = await new Promise<Error>((resolve) => {
				service.createModelsPoll(100).subscribe({ next: () => {}, error: resolve });
				vi.advanceTimersByTime(0);
			});

			expect(error.message).toBe('Network failure');
		});
	});
});
