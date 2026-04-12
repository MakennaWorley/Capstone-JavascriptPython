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
			<span style={{ color: 'rgba(255, 255, 255, 0.6)', fontSize: '0.9rem' }}>
				Failed to load datasets. Please check your connection and refresh the page.
			</span>
		);
	}

	return (
		<FormControl fullWidth size="small" disabled={disabled}>
			<InputLabel
				sx={{
					color: 'rgba(255, 255, 255, 0.7)',
					'&.Mui-focused': { color: '#646cff' }
				}}
			>
				Choose a dataset
			</InputLabel>
			<Select
				value={selected}
				onChange={(e) => onSelect(e.target.value)}
				label="Choose a dataset"
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
				{datasets.map((d) => (
					<MenuItem key={d} value={d}>
						{d}
					</MenuItem>
				))}
			</Select>
		</FormControl>
	);
}
