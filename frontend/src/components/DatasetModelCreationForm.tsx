import { Alert, Box, Button, Checkbox, FormControlLabel, Paper, Stack, TextField, Typography } from '@mui/material';
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
	debugMode?: boolean;
	onSuccess?: () => void;
	onSuccessNotification?: (message: string) => void;
};

// ---------- defaults ----------
const DEFAULTS: SimConfig = {
	name: '',
	sequence_length: 100,
	n_generations: 5,
	samples_per_generation: 50
};

export default function DatasetModelCreationForm({
	apiBase,
	xApiKey,
	endpoint = '/api/create/data',
	debugMode = false,
	onSuccess,
	onSuccessNotification
}: Props) {
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

			setResponseJson(maybeJson ?? text);
			if (onSuccessNotification) {
				onSuccessNotification('Dataset created successfully!');
			}
			if (onSuccess) {
				onSuccess();
			}
		} catch (err) {
			console.error(err);
			setStatus('Network error');
		} finally {
			setSending(false);
		}
	}

	return (
		<Box component="form" onSubmit={submit} sx={{ display: 'grid', gap: '1.5rem', maxWidth: 720, pt: 2, pb: 2 }}>
			{/* BASIC */}
			<Box>
				<Typography variant="h6" sx={{ mb: 2, color: 'text.primary' }}>
					Basic Settings
				</Typography>
				<Stack spacing={2}>
					<TextField
						label="Dataset name (alphanumeric, no spaces)"
						value={cfg.name}
						onChange={(e) => update('name', e.target.value)}
						placeholder="mydataset01"
						fullWidth
						required
						variant="outlined"
						size="small"
						sx={{
							'& .MuiOutlinedInput-root': {
								color: 'text.primary',
								'& fieldset': { borderColor: '#452ee4', borderWidth: '2px' },
								'&:hover fieldset': { borderColor: '#241291', borderWidth: '2px' },
								'&.Mui-focused fieldset': { borderColor: '#452ee4', borderWidth: '2px' }
							},
							'& .MuiInputBase-input::placeholder': { color: 'text.disabled', opacity: 1 },
							'& .MuiInputLabel-root': { color: 'text.secondary' },
							'& .MuiInputLabel-root.Mui-focused': { color: '#452ee4' }
						}}
					/>
					<FormControlLabel
						control={
							<Checkbox
								checked={advanced}
								onChange={(e) => setAdvanced(e.target.checked)}
								sx={{
									color: '#452ee4',
									'&.Mui-checked': { color: '#452ee4' }
								}}
							/>
						}
						label="Advanced Settings (scale individuals)"
						sx={{ color: 'text.primary' }}
					/>
				</Stack>
			</Box>

			{/* ADVANCED */}
			{advanced && (
				<Box>
					<Typography variant="h6" sx={{ mb: 1, color: 'text.primary' }}>
						Advanced Settings <span style={{ fontSize: '0.75rem', fontWeight: 'normal' }}>(optional)</span>
					</Typography>
					<Typography variant="body2" sx={{ mb: 2, color: 'text.secondary', lineHeight: 1.6 }}>
						These settings control how large and deep the simulated family is. Larger values produce richer, more realistic data but take
						longer to generate. The defaults are a good starting point for most experiments.
					</Typography>
					<Stack spacing={2}>
						<Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 2 }}>
							<TextField
								label="Sequence Length"
								type="number"
								value={Number.isFinite(cfg.sequence_length) ? cfg.sequence_length : ''}
								placeholder="100"
								onChange={(e) => update('sequence_length', Number(e.target.value))}
								inputProps={{ min: 1, step: 1 }}
								helperText="Number of genomic sites (SNP positions) simulated per individual. More sites = more features for the model, but slower generation. Max 1,000."
								variant="outlined"
								size="small"
								sx={{
									'& .MuiOutlinedInput-root': {
										color: 'text.primary',
										'& fieldset': { borderColor: '#452ee4', borderWidth: '2px' },
										'&:hover fieldset': { borderColor: '#241291', borderWidth: '2px' },
										'&.Mui-focused fieldset': { borderColor: '#452ee4', borderWidth: '2px' }
									},
									'& .MuiInputBase-input::placeholder': { color: 'text.disabled', opacity: 1 },
									'& .MuiInputLabel-root': { color: 'text.secondary' },
									'& .MuiInputLabel-root.Mui-focused': { color: '#452ee4' }
								}}
							/>
							<TextField
								label="Number of generations"
								type="number"
								value={Number.isFinite(cfg.n_generations) ? cfg.n_generations : ''}
								placeholder="5"
								onChange={(e) => update('n_generations', e.target.value === '' ? (NaN as any) : Number(e.target.value))}
								inputProps={{ min: 1, step: 1 }}
								helperText="How many parent-to-child generations the family spans. More generations = deeper pedigree structure. Max 10."
								variant="outlined"
								size="small"
								sx={{
									'& .MuiOutlinedInput-root': {
										color: 'text.primary',
										'& fieldset': { borderColor: '#452ee4', borderWidth: '2px' },
										'&:hover fieldset': { borderColor: '#241291', borderWidth: '2px' },
										'&.Mui-focused fieldset': { borderColor: '#452ee4', borderWidth: '2px' }
									},
									'& .MuiInputBase-input::placeholder': { color: 'text.disabled', opacity: 1 },
									'& .MuiInputLabel-root': { color: 'text.secondary' },
									'& .MuiInputLabel-root.Mui-focused': { color: '#452ee4' }
								}}
							/>
							<TextField
								label="Individuals per generation"
								type="number"
								value={Number.isFinite(cfg.samples_per_generation) ? cfg.samples_per_generation : ''}
								placeholder="50"
								onChange={(e) => update('samples_per_generation', e.target.value === '' ? (NaN as any) : Number(e.target.value))}
								inputProps={{ min: 1, step: 1 }}
								helperText="How many diploid individuals are simulated in each generation. Combined with generations, this sets the total family size. Total must not exceed 1,000."
								variant="outlined"
								size="small"
								sx={{
									'& .MuiOutlinedInput-root': {
										color: 'text.primary',
										'& fieldset': { borderColor: '#452ee4', borderWidth: '2px' },
										'&:hover fieldset': { borderColor: '#241291', borderWidth: '2px' },
										'&.Mui-focused fieldset': { borderColor: '#452ee4', borderWidth: '2px' }
									},
									'& .MuiInputBase-input::placeholder': { color: 'text.disabled', opacity: 1 },
									'& .MuiInputLabel-root': { color: 'text.secondary' },
									'& .MuiInputLabel-root.Mui-focused': { color: '#452ee4' }
								}}
							/>
						</Box>
						<Typography variant="body2" sx={{ color: 'text.secondary', mt: 1 }}>
							Total individuals: <strong>{derivedTotal ?? '—'}</strong>{' '}
							{derivedTotal !== undefined && (
								<span style={{ marginLeft: '0.5rem', opacity: 0.75 }}>
									(
									{derivedTotal <= 250
										? 'small dataset — fast, good for testing'
										: derivedTotal <= 1000
											? 'medium dataset — balanced speed and coverage'
											: 'large dataset — may take a while to generate'}
									)
								</span>
							)}{' '}
						</Typography>
					</Stack>
				</Box>
			)}

			{/* SUBMIT BUTTON */}
			<Button
				type="submit"
				disabled={sending}
				variant="contained"
				fullWidth
				sx={{
					backgroundColor: '#452ee4',
					padding: '0.75rem',
					fontSize: '1rem',
					'&:hover': { backgroundColor: '#241291' },
					'&:disabled': { backgroundColor: '#555', color: 'rgba(255, 255, 255, 0.5)' }
				}}
			>
				{sending ? 'Generating...' : 'Generate Data'}
			</Button>

			<LoadingProgress isLoading={sending} message="Generating your data..." />

			{/* STATUS MESSAGE - ERRORS ONLY */}
			{status && status !== 'Dataset created successfully!' && (
				<Alert
					severity="error"
					sx={{
						backgroundColor: 'rgba(255, 107, 107, 0.1)',
						color: '#ff6b6b',
						border: '1px solid #ff6b6b'
					}}
				>
					{status}
				</Alert>
			)}

			{/* DEBUG JSON OUTPUT */}
			{debugMode && responseJson && (
				<Paper sx={{ p: 2, bgcolor: 'background.default', border: '2px solid #452ee4', borderRadius: '4px' }}>
					<Typography variant="caption" sx={{ color: '#452ee4', display: 'block', mb: 1 }}>
						Debug: Response
					</Typography>
					<Box
						component="pre"
						sx={{
							whiteSpace: 'pre-wrap',
							wordWrap: 'break-word',
							color: 'text.primary',
							fontSize: '0.75rem',
							overflow: 'auto',
							maxHeight: '300px'
						}}
					>
						{typeof responseJson === 'string' ? responseJson : JSON.stringify(responseJson, null, 2)}
					</Box>
				</Paper>
			)}
		</Box>
	);
}
