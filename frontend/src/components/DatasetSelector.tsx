import { MenuItem, TextField } from '@mui/material';

type DatasetSelectorProps = {
	datasets: string[];
	selected: string;
	onSelect: (dataset: string) => void;
	disabled?: boolean;
};

export default function DatasetSelector({ datasets, selected, onSelect, disabled = false }: DatasetSelectorProps) {
	if (datasets.length === 0) {
		return <span className="error-inline">Failed to load datasets. Please check your connection and refresh the page.</span>;
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
			className="purple-select"
		>
			<MenuItem value="" disabled>
				Select…
			</MenuItem>
			{[...datasets]
				.sort((a, b) => a.localeCompare(b))
				.map((d) => (
					<MenuItem key={d} value={d}>
						{d}
					</MenuItem>
				))}
		</TextField>
	);
}
