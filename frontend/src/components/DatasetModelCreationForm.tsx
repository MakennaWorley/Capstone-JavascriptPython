import React, { useMemo, useState } from 'react';
import LoadingProgress from './LoadingProgress.js';

// ---------- helpers ----------
const ALNUM_RE = /^[A-Za-z0-9]+$/;

function isAlnumNoWhitespace(s: string) {
	return ALNUM_RE.test(s);
}

// ---------- Security Constants (Sync with Python) ----------
const MAX_USER_SAMPLES = 1000;
const MAX_USER_LENGTH = 1000;
const MAX_USER_GENERATIONS = 10;

// ---------- types ----------
type SimConfig = {
	// basic
	name: string;

	// advanced (scaling)
	sequence_length: number;
	n_generations: number;
	samples_per_generation: number;
};

type GenerateRequest = {
	params: { name: string } & Partial<SimConfig> & { n_diploid_samples?: number };
};

type Props = {
	apiBase: string;
	xApiKey: string;
	endpoint?: string;
};

// ---------- defaults ----------
const DEFAULTS: SimConfig = {
	name: '',
	sequence_length: 5,
	n_generations: 0,
	samples_per_generation: 50
};

export default function DatasetModelCreationForm({ apiBase, xApiKey, endpoint = '/api/create/data' }: Props) {
	const [advanced, setAdvanced] = useState(false);
	const [sending, setSending] = useState(false);
	const [status, setStatus] = useState('');
	const [responseJson, setResponseJson] = useState<any>(null);

	const [cfg, setCfg] = useState<SimConfig>(DEFAULTS);

	function update<K extends keyof SimConfig>(key: K, value: SimConfig[K]) {
		setCfg((prev) => ({ ...prev, [key]: value }));
	}

	const derivedTotal =
		Number.isFinite(cfg.n_generations) && Number.isFinite(cfg.samples_per_generation)
			? cfg.n_generations * cfg.samples_per_generation
			: undefined;

	// ---------- validation ----------
	const errors = useMemo(() => {
		const e: string[] = [];

		if (!cfg.name.trim()) e.push('Dataset name is required.');
		if (cfg.name.trim() && !isAlnumNoWhitespace(cfg.name.trim())) {
			e.push('Dataset name must be alphanumeric only (no spaces).');
		}

		if (advanced) {
			if (!Number.isFinite(cfg.sequence_length) || !Number.isInteger(cfg.sequence_length) || cfg.sequence_length <= 0) {
				e.push('Sequence length must be a positive number.');
			}
			if (!Number.isFinite(cfg.n_generations) || !Number.isInteger(cfg.n_generations) || cfg.n_generations <= 0) {
				e.push('Number of generations must be a positive integer.');
			}
			if (!Number.isFinite(cfg.samples_per_generation) || !Number.isInteger(cfg.samples_per_generation) || cfg.samples_per_generation <= 0) {
				e.push('Individuals per generation must be a positive integer.');
			}

			if (cfg.sequence_length > MAX_USER_LENGTH) e.push(`Sequence length cannot exceed ${MAX_USER_LENGTH}.`);
			if (cfg.n_generations > MAX_USER_GENERATIONS) e.push(`Generations cannot exceed ${MAX_USER_GENERATIONS}.`);
			if (cfg.n_generations * cfg.samples_per_generation > MAX_USER_SAMPLES)
				e.push(
					`Total individuals cannot exceed ${MAX_USER_SAMPLES}. Please lower either number of generations or individuals per generation.`
				);
		}

		return e;
	}, [cfg, advanced]);

	// ---------- submit ----------
	async function submit(e: React.FormEvent) {
		e.preventDefault();
		if (errors.length) return;

		setSending(true);
		setStatus('');
		setResponseJson(null);

		try {
			const params: GenerateRequest['params'] = {
				name: cfg.name.trim()
			};

			// Only include parameters if Advanced is enabled
			if (advanced) {
				params.sequence_length = cfg.sequence_length;
				params.n_generations = cfg.n_generations;
				params.samples_per_generation = cfg.samples_per_generation;
				params.n_diploid_samples = cfg.n_generations * cfg.samples_per_generation;
			}

			const payload: GenerateRequest = { params };

			const r = await fetch(`${apiBase}${endpoint}`, {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json',
					'X-API-Key': xApiKey
				},
				body: JSON.stringify(payload)
			});

			const text = await r.text().catch(() => '');
			let maybeJson: any = null;
			try {
				maybeJson = text ? JSON.parse(text) : null;
			} catch {}

			if (!r.ok) {
				// Extract error message from response if available
				let errorMessage = `Error ${r.status}`;
				if (maybeJson?.message) {
					errorMessage = maybeJson.message;
				} else if (typeof text === 'string' && text) {
					errorMessage = text;
				}
				setStatus(errorMessage);
				setResponseJson(maybeJson ?? text);
				return;
			}

			setStatus('Success!');
			setResponseJson(maybeJson ?? text);
		} catch (err) {
			console.error(err);
			setStatus('Network error');
		} finally {
			setSending(false);
		}
	}

	return (
		<form onSubmit={submit} style={{ display: 'grid', gap: '0.9rem', maxWidth: 720 }}>
			<h2>Create Dataset</h2>

			{/* BASIC */}
			<fieldset style={{ padding: '1rem' }}>
				<legend>Basic</legend>

				<label style={{ display: 'grid', gap: 6 }}>
					Dataset name (alphanumeric, no spaces)
					<input
						value={cfg.name}
						onChange={(e) => update('name', e.target.value)}
						placeholder="mydataset01"
						style={{ padding: '0.5rem' }}
					/>
				</label>

				<label style={{ display: 'flex', gap: 8, marginTop: 12 }}>
					<input type="checkbox" checked={advanced} onChange={(e) => setAdvanced(e.target.checked)} />
					Advanced Settings (scale individuals)
				</label>
			</fieldset>

			{/* ADVANCED */}
			{advanced && (
				<fieldset style={{ padding: '1rem' }}>
					<legend>
						Advanced <span style={{ fontSize: '0.8rem', fontWeight: 'normal' }}>(optional)</span>
					</legend>

					<div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
						<label>
							Sequence Length
							<input
								type="number"
								value={Number.isFinite(cfg.sequence_length) ? cfg.sequence_length : ''}
								placeholder="100"
								onChange={(e) => update('sequence_length', Number(e.target.value))}
								min={1}
								step={1}
								style={{ padding: '0.5rem' }}
							/>
						</label>

						<label>
							Number of generations
							<input
								type="number"
								value={Number.isFinite(cfg.n_generations) ? cfg.n_generations : ''}
								placeholder="5"
								onChange={(e) => update('n_generations', e.target.value === '' ? (NaN as any) : Number(e.target.value))}
								min={1}
								step={1}
								style={{ padding: '0.5rem' }}
							/>
						</label>

						<label>
							Individuals per generation
							<input
								type="number"
								value={Number.isFinite(cfg.samples_per_generation) ? cfg.samples_per_generation : ''}
								placeholder="50"
								onChange={(e) => update('samples_per_generation', e.target.value === '' ? (NaN as any) : Number(e.target.value))}
								min={1}
								step={1}
								style={{ padding: '0.5rem' }}
							/>
						</label>
					</div>

					<p style={{ fontSize: '0.9rem', marginTop: 10 }}>
						Total individuals: <strong>{derivedTotal ?? '—'}</strong>
					</p>
				</fieldset>
			)}

			<button type="submit" disabled={sending || errors.length > 0} style={{ padding: '0.75rem' }}>
				{sending ? 'Generating...' : 'Generate Data'}
			</button>

			<LoadingProgress isLoading={sending} message="Generating your data..." />

			{status && <p>{status}</p>}

			{responseJson && (
				<pre style={{ whiteSpace: 'pre-wrap', padding: '0.75rem', border: '1px solid #ddd' }}>
					{typeof responseJson === 'string' ? responseJson : JSON.stringify(responseJson, null, 2)}
				</pre>
			)}
		</form>
	);
}
