import { FormControl, InputLabel, MenuItem, Select } from '@mui/material';

type DatasetSelectorProps = {
	datasets: string[];
	selected: string;
	onSelect: (dataset: string) => void;
	disabled?: boolean;
};

export default function DatasetSelector({ datasets, selected, onSelect, disabled = false }: DatasetSelectorProps) {
	if (datasets.length === 0) {
		return (
			<span style={{ color: 'inherit', fontSize: '0.9rem' }}>Failed to load datasets. Please check your connection and refresh the page.</span>
		);
	}

	return (
		<FormControl fullWidth size="small" disabled={disabled}>
			<InputLabel
				sx={{
					color: 'text.secondary',
					'&.Mui-focused': { color: '#452ee4' }
				}}
			>
				Choose a dataset
			</InputLabel>
			<Select
				value={selected}
				onChange={(e) => onSelect(e.target.value)}
				label="Choose a dataset"
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
				{datasets.map((d) => (
					<MenuItem key={d} value={d}>
						{d}
					</MenuItem>
				))}
			</Select>
		</FormControl>
	);
}
