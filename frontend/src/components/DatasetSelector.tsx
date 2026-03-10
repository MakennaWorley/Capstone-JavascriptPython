type DatasetSelectorProps = {
	datasets: string[];
	selected: string;
	onSelect: (dataset: string) => void;
	disabled?: boolean;
};

export default function DatasetSelector({ datasets, selected, onSelect, disabled = false }: DatasetSelectorProps) {
	if (datasets.length === 0) {
		return (
			<p style={{ opacity: 0.8 }}>
				Click <b>List Datasets</b> first.
			</p>
		);
	}

	return (
		<label>
			<span style={{ display: 'block', fontSize: '0.9rem', marginBottom: '0.25rem' }}>Choose a dataset</span>
			<select value={selected} onChange={(e) => onSelect(e.target.value)} disabled={disabled} style={{ padding: '0.4rem', minWidth: '260px' }}>
				<option value="" disabled>
					Select…
				</option>
				{datasets.map((d) => (
					<option key={d} value={d}>
						{d}
					</option>
				))}
			</select>
		</label>
	);
}
