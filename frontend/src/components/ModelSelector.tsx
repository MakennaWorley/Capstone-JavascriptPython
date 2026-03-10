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
			<p style={{ opacity: 0.8 }}>
				Click <b>List Models</b> first.
			</p>
		);
	}

	// Create a unique key for each model
	const getModelKey = (model: Model) => `${model.model_name}::${model.model_type}`;
	const selectedKey = selected ? getModelKey(selected) : '';

	const handleChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
		const key = e.target.value;
		if (!key) {
			onSelect(null);
			return;
		}

		const model = models.find((m) => getModelKey(m) === key);
		onSelect(model || null);
	};

	return (
		<label>
			<span style={{ display: 'block', fontSize: '0.9rem', marginBottom: '0.25rem' }}>Choose a model</span>
			<select value={selectedKey} onChange={handleChange} disabled={disabled} style={{ padding: '0.4rem', minWidth: '260px' }}>
				<option value="" disabled>
					Select…
				</option>
				{models.map((model) => {
					const key = getModelKey(model);
					return (
						<option key={key} value={key}>
							{model.model_name} ({model.model_type})
						</option>
					);
				})}
			</select>
		</label>
	);
}
