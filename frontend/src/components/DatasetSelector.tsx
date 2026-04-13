import { MenuItem, TextField } from '@mui/material';

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
		<TextField
			select
			fullWidth
			size="small"
			disabled={disabled}
			label="Choose a dataset"
			value={selected}
			onChange={(e) => onSelect(e.target.value)}
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
			{[...datasets].sort((a, b) => a.localeCompare(b)).map((d) => (
				<MenuItem key={d} value={d}>
					{d}
				</MenuItem>
			))}
		</TextField>
	);
}
