import { useTheme } from '@mui/material/styles';

type Model = {
	model_name: string;
	model_type: string;
};

type ModelDashboardProps = {
	model: Model;
};

const MODEL_TYPE_INFO: Record<string, { label: string; description: string; strengths: string[]; use_case: string }> = {
	dnn_dosage: {
		label: 'Deep Neural Network (Dosage)',
		description:
			'A fully connected deep neural network trained to predict genotype dosage values. Uses multiple hidden layers to learn complex non-linear relationships in genomic data.',
		strengths: ['Captures non-linear patterns', 'Scales well with data size', 'Flexible architecture'],
		use_case: 'Best suited for datasets with large numbers of individuals where complex interactions between loci are expected.'
	},
	multi_log_regression: {
		label: 'Multinomial Logistic Regression',
		description:
			'A classical statistical model that predicts class probabilities for each genotype class (0, 1, 2). Trains a separate set of weights per class using a softmax output.',
		strengths: ['Interpretable coefficients', 'Fast to train', 'Good baseline model'],
		use_case: 'Works well as a fast baseline or when the relationship between features and genotype is approximately linear.'
	},
	hmm_dosage: {
		label: 'Hidden Markov Model (Dosage)',
		description:
			'A probabilistic model that captures sequential dependencies along the genome. Models each locus as an emission from a hidden genetic state.',
		strengths: ['Explicitly models genomic sequence structure', 'Probabilistic outputs', 'Interpretable hidden states'],
		use_case: 'Ideal when linkage disequilibrium and sequential structure along chromosomes is important to capture.'
	},
	gnn_dosage: {
		label: 'Graph Neural Network (Dosage)',
		description:
			'A graph-based deep learning model that operates over pedigree graphs. Message passing between connected individuals allows the model to leverage family-level genetic information.',
		strengths: ['Leverages pedigree structure', 'Propagates information across relatives', 'Learns relational patterns'],
		use_case:
			'Most powerful when pedigree relationships are available and individuals share genetic material with close relatives in the dataset.'
	},
	bayesian: {
		label: 'Bayesian Classifier',
		description:
			'A probabilistic model that estimates posterior genotype probabilities using prior knowledge of allele frequencies and observed data likelihoods.',
		strengths: ['Principled uncertainty quantification', 'Naturally incorporates prior knowledge', 'Robust to small samples'],
		use_case: 'A strong choice when prior allele frequency information is available or when calibrated confidence estimates are critical.'
	}
};

const FALLBACK_INFO = {
	label: 'Custom Model',
	description: 'This model type does not have a built-in description. Refer to the model documentation for details.',
	strengths: ['User-defined architecture'],
	use_case: 'Application-specific. Consult the model configuration for details.'
};

export default function ModelDashboard({ model }: ModelDashboardProps) {
	const theme = useTheme();
	const isDark = theme.palette.mode === 'dark';
	const info = MODEL_TYPE_INFO[model.model_type] ?? FALLBACK_INFO;

	return (
		<div style={{ marginTop: '1.25rem', padding: '1.5rem', backgroundColor: theme.palette.background.default, borderRadius: '8px' }}>
			<h3 style={{ marginTop: 0 }}>Model Dashboard</h3>

			{/* Identity */}
			<div
				style={{
					display: 'grid',
					gridTemplateColumns: '1fr 1fr',
					gap: '1rem',
					marginBottom: '1.5rem'
				}}
			>
				<div
					style={{
						padding: '1rem',
						backgroundColor: theme.palette.background.default,
						borderRadius: '6px',
						border: `1px solid ${theme.palette.divider}`
					}}
				>
					<p
						style={{
							margin: 0,
							fontSize: '0.75rem',
							color: theme.palette.text.secondary,
							textTransform: 'uppercase',
							letterSpacing: '0.05em'
						}}
					>
						Model Name
					</p>
					<p style={{ margin: '0.4rem 0 0 0', fontSize: '1.1rem', fontWeight: 'bold' }}>{model.model_name}</p>
				</div>
				<div
					style={{
						padding: '1rem',
						backgroundColor: theme.palette.background.default,
						borderRadius: '6px',
						border: `1px solid ${theme.palette.divider}`
					}}
				>
					<p
						style={{
							margin: 0,
							fontSize: '0.75rem',
							color: theme.palette.text.secondary,
							textTransform: 'uppercase',
							letterSpacing: '0.05em'
						}}
					>
						Model Type
					</p>
					<p style={{ margin: '0.4rem 0 0 0', fontSize: '1.1rem', fontWeight: 'bold' }}>{info.label}</p>
				</div>
			</div>

			{/* Description */}
			<div style={{ marginBottom: '1.5rem' }}>
				<h4 style={{ marginTop: 0, marginBottom: '0.5rem' }}>About this Model</h4>
				<p style={{ margin: 0, lineHeight: '1.6', opacity: 0.85 }}>{info.description}</p>
			</div>

			{/* Strengths */}
			<div style={{ marginBottom: '1.5rem' }}>
				<h4 style={{ marginTop: 0, marginBottom: '0.5rem' }}>Strengths</h4>
				<ul style={{ margin: 0, paddingLeft: '1.4rem', lineHeight: '1.8', opacity: 0.85 }}>
					{info.strengths.map((s) => (
						<li key={s}>{s}</li>
					))}
				</ul>
			</div>

			{/* Use case */}
			<div
				style={{
					padding: '1rem',
					backgroundColor: isDark ? '#0d1f0d' : '#e8f5e9',
					borderRadius: '6px',
					border: `1px solid ${isDark ? '#1e4d1e' : '#66bb6a'}`
				}}
			>
				<p
					style={{
						margin: 0,
						fontSize: '0.75rem',
						color: isDark ? '#6fcf6f' : '#2e7d32',
						textTransform: 'uppercase',
						letterSpacing: '0.05em',
						marginBottom: '0.4rem'
					}}
				>
					Recommended Use Case
				</p>
				<p style={{ margin: 0, lineHeight: '1.6' }}>{info.use_case}</p>
			</div>
		</div>
	);
}
