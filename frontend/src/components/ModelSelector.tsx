import { FormControl, InputLabel, MenuItem, Select } from '@mui/material';

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
		<FormControl fullWidth size="small" disabled={disabled}>
			<InputLabel
				sx={{
					color: 'text.secondary',
					'&.Mui-focused': { color: '#452ee4' }
				}}
			>
				Choose a model
			</InputLabel>
			<Select
				value={selectedKey}
				onChange={(e) => handleChange(e.target.value)}
				label="Choose a model"
				sx={{
					color: 'text.primary',
					'& .MuiOutlinedInput-notchedOutline': { borderColor: '#452ee4', borderWidth: '2px' },
					'&:hover .MuiOutlinedInput-notchedOutline': { borderColor: '#241291', borderWidth: '2px' },
					'&.Mui-focused .MuiOutlinedInput-notchedOutline': { borderColor: '#452ee4', borderWidth: '2px' },
					'& .MuiSvgIcon-root': { color: '#452ee4' }
				}}
			>
				<MenuItem value="" disabled>
					Select…
				</MenuItem>
				{models.map((model) => {
					const key = getModelKey(model);
					return (
						<MenuItem key={key} value={key}>
							{model.model_name} ({model.model_type})
						</MenuItem>
					);
				})}
			</Select>
		</FormControl>
	);
}
