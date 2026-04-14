import { Alert, Box, Button, Checkbox, FormControlLabel, Paper, Stack, TextField, Typography } from '@mui/material';
import { useMemo, useState } from 'react';
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
	sequenceLength: number;
	nGenerations: number;
	samplesPerGeneration: number;
};

type GenerateRequest = {
	params: {
		name: string;
		sequence_length?: number;
		n_generations?: number;
		samples_per_generation?: number;
		n_diploid_samples?: number;
	};
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
	sequenceLength: 100,
	nGenerations: 5,
	samplesPerGeneration: 50
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
	const [submitted, setSubmitted] = useState(false);

	const [cfg, setCfg] = useState<SimConfig>(DEFAULTS);

	function update<K extends keyof SimConfig>(key: K, value: SimConfig[K]) {
		setCfg((prev) => ({ ...prev, [key]: value }));
	}

	const derivedTotal =
		Number.isFinite(cfg.nGenerations) && Number.isFinite(cfg.samplesPerGeneration)
			? cfg.nGenerations * cfg.samplesPerGeneration
			: undefined;

	// ---------- validation ----------
	const errors = useMemo(() => {
		const e: string[] = [];

		if (!cfg.name.trim()) e.push('Dataset name is required.');
		if (cfg.name.trim() && !isAlnumNoWhitespace(cfg.name.trim())) {
			e.push('Dataset name must be alphanumeric only (no spaces).');
		}

		if (advanced) {
			if (!Number.isFinite(cfg.sequenceLength) || !Number.isInteger(cfg.sequenceLength) || cfg.sequenceLength <= 0) {
				e.push('Sequence length must be a positive number.');
			}
			if (!Number.isFinite(cfg.nGenerations) || !Number.isInteger(cfg.nGenerations) || cfg.nGenerations <= 0) {
				e.push('Number of generations must be a positive integer.');
			}
			if (!Number.isFinite(cfg.samplesPerGeneration) || !Number.isInteger(cfg.samplesPerGeneration) || cfg.samplesPerGeneration <= 0) {
				e.push('Individuals per generation must be a positive integer.');
			}

			if (cfg.sequenceLength > MAX_USER_LENGTH) e.push(`Sequence length cannot exceed ${MAX_USER_LENGTH}.`);
			if (cfg.nGenerations > MAX_USER_GENERATIONS) e.push(`Generations cannot exceed ${MAX_USER_GENERATIONS}.`);
			if (cfg.nGenerations * cfg.samplesPerGeneration > MAX_USER_SAMPLES)
				e.push(
					`Total individuals cannot exceed ${MAX_USER_SAMPLES}. Please lower either number of generations or individuals per generation.`
				);
		}

		return e;
	}, [cfg, advanced]);

	// ---------- submit ----------
	function handleSubmit() {
		setSubmitted(true);

		if (errors.length > 0) return;

		performSubmit();
	}

	async function performSubmit() {
		setSending(true);
		setStatus('');
		setResponseJson(null);

		try {
			const params: GenerateRequest['params'] = {
				name: cfg.name.trim()
			};

			// Only include parameters if Advanced is enabled
			if (advanced) {
				params.sequence_length = cfg.sequenceLength;
				params.n_generations = cfg.nGenerations;
				params.samples_per_generation = cfg.samplesPerGeneration;
				params.n_diploid_samples = cfg.nGenerations * cfg.samplesPerGeneration;
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
		<Box component="form" onSubmit={(e) => { e.preventDefault(); handleSubmit(); }} sx={{ display: 'grid', gap: '1.5rem', maxWidth: 720, pt: 2, pb: 2 }}>
			{/* AUTO-DELETE NOTICE */}
			<Alert severity="info">
				Datasets are automatically deleted within 24 hours of creation.
			</Alert>

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
					<Typography variant="h6" sx={{ mb: 2, color: 'text.primary' }}>
						Advanced Settings <span style={{ fontSize: '0.75rem', fontWeight: 'normal' }}>(optional)</span>
					</Typography>
					<Stack spacing={2}>
						<Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 2 }}>
							<TextField
								label="Sequence Length"
								type="number"
								value={Number.isFinite(cfg.sequenceLength) ? cfg.sequenceLength : ''}
								placeholder="100"
								onChange={(e) => update('sequenceLength', Number(e.target.value))}
								inputProps={{ min: 1, step: 1 }}
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
								value={Number.isFinite(cfg.nGenerations) ? cfg.nGenerations : ''}
								placeholder="5"
								onChange={(e) => update('nGenerations', e.target.value === '' ? (NaN as any) : Number(e.target.value))}
								inputProps={{ min: 1, step: 1 }}
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
								value={Number.isFinite(cfg.samplesPerGeneration) ? cfg.samplesPerGeneration : ''}
								placeholder="50"
								onChange={(e) => update('samplesPerGeneration', e.target.value === '' ? (NaN as any) : Number(e.target.value))}
								inputProps={{ min: 1, step: 1 }}
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
							Total individuals: <strong>{derivedTotal ?? '—'}</strong>
						</Typography>
					</Stack>
				</Box>
			)}

			{/* SUBMIT BUTTON */}
			<Button
				type="button"
				disabled={sending}
				onClick={handleSubmit}
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
				{sending ? 'Generating Dataset...' : 'Generate Dataset'}
			</Button>

			<LoadingProgress isLoading={sending} message="Generating your data..." />

			{/* VALIDATION ERRORS */}
			{submitted && errors.length > 0 && (
				<Alert
					severity="error"
					sx={{
						backgroundColor: 'rgba(255, 107, 107, 0.1)',
						color: '#ff6b6b',
						border: '1px solid #ff6b6b'
					}}
				>
					{errors.map((error, i) => (
						<div key={i}>{error}</div>
					))}
				</Alert>
			)}

			{/* STATUS MESSAGE - ERRORS ONLY */}
			{status && (
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
