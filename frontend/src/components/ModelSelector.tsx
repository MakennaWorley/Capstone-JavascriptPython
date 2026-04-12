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
			<span style={{ color: 'rgba(255, 255, 255, 0.6)', fontSize: '0.9rem' }}>
				Failed to load models. Please check your connection and refresh the page.
			</span>
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
					color: 'rgba(255, 255, 255, 0.7)',
					'&.Mui-focused': { color: '#646cff' }
				}}
			>
				Choose a model
			</InputLabel>
			<Select
				value={selectedKey}
				onChange={(e) => handleChange(e.target.value)}
				label="Choose a model"
				sx={{
					color: '#fff',
					'& .MuiOutlinedInput-notchedOutline': { borderColor: '#646cff' },
					'&:hover .MuiOutlinedInput-notchedOutline': { borderColor: '#747eff' },
					'&.Mui-focused .MuiOutlinedInput-notchedOutline': { borderColor: '#646cff' },
					'& .MuiSvgIcon-root': { color: '#646cff' }
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
