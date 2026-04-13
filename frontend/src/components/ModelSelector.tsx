import { MenuItem, TextField } from '@mui/material';

type Model = {
	model_name: string;
	model_type: string;
};

type ModelSelectorProps = {
	models: Model[];
	selected: { model_name: string; model_type: string } | null;
	onSelect: (model: Model | null) => void;
	disabled?: boolean;
};

const MODEL_TYPE_SHORT_NAMES: Record<string, string> = {
	bayes_softmax3: 'Bayesian Inference',
	multi_log_regression: 'Multinomial Logistic Regression',
	hmm_dosage: 'Hidden Markov Model',
	dnn_dosage: 'Deep Neural Network',
	gnn_dosage: 'Graph Neural Network'
};

function capitalize(s: string): string {
	if (!s) return s;
	return s.charAt(0).toUpperCase() + s.slice(1);
}

export default function ModelSelector({ models, selected, onSelect, disabled = false }: ModelSelectorProps) {
	if (models.length === 0) {
		return (
			<span style={{ color: 'inherit', fontSize: '0.9rem' }}>Failed to load models. Please check your connection and refresh the page.</span>
		);
	}

	// Create a unique key for each model
	const getModelKey = (model: Model) => `${model.model_name}::${model.model_type}`;
	const selectedKey = selected ? getModelKey(selected) : '';

	const handleChange = (value: string) => {
		if (!value) {
			onSelect(null);
			return;
		}

		const model = models.find((m) => getModelKey(m) === value);
		onSelect(model || null);
	};

	return (
		<TextField
			select
			fullWidth
			size="small"
			disabled={disabled}
			label="Choose a model"
			value={selectedKey}
			onChange={(e) => handleChange(e.target.value)}
			slotProps={{
				inputLabel: {
					sx: { color: 'text.secondary', '&.Mui-focused': { color: '#452ee4' } }
				}
			}}
			sx={{
				'& .MuiOutlinedInput-notchedOutline': { borderColor: '#452ee4', borderWidth: '2px' },
				'&:hover .MuiOutlinedInput-notchedOutline': { borderColor: '#241291', borderWidth: '2px' },
				'& .Mui-focused .MuiOutlinedInput-notchedOutline': { borderColor: '#452ee4', borderWidth: '2px' },
				'& .MuiSvgIcon-root': { color: '#452ee4' },
				'& .MuiInputBase-input': { color: 'text.primary' }
			}}
		>
			<MenuItem value="" disabled>
				Select…
			</MenuItem>
			{[...models]
				.sort((a, b) => {
					const labelA = `${capitalize(a.model_name)} ${MODEL_TYPE_SHORT_NAMES[a.model_type] ?? a.model_type}`;
					const labelB = `${capitalize(b.model_name)} ${MODEL_TYPE_SHORT_NAMES[b.model_type] ?? b.model_type}`;
					return labelA.localeCompare(labelB);
				})
				.map((model) => {
					const key = getModelKey(model);
					const shortName = MODEL_TYPE_SHORT_NAMES[model.model_type] ?? model.model_type;
					const sizeName = capitalize(model.model_name);
					return (
						<MenuItem key={key} value={key}>
							{sizeName} {shortName}
						</MenuItem>
					);
				})}
		</TextField>
	);
}
