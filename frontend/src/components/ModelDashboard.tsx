import { FormControlLabel, Switch } from '@mui/material';
import { useTheme } from '@mui/material/styles';
import { useState } from 'react';

type Model = {
	model_name: string;
	model_type: string;
};

type ModelDashboardProps = {
	model: Model;
};

const MODEL_TYPE_INFO: Record<
	string,
	{
		label: string;
		simple_description: string;
		simple_strengths: string[];
		simple_use_case: string;
		description: string;
		strengths: string[];
		use_case: string;
	}
> = {
	dnn_dosage: {
		label: 'Deep Neural Network',
		simple_description:
			'Our most capable pattern-recognition model. It independently discovers hidden connections across your entire genome — no manual feature engineering required. The more data you give it, the smarter it gets.',
		simple_strengths: ['Finds patterns humans would not think to look for', 'Gets more accurate as your dataset grows', 'Handles complex, high-dimensional genetic data well'],
		simple_use_case: 'Best for large datasets (100+ individuals) where you want the highest possible accuracy and are not pressed for training time.',
		description:
			'A fully connected deep neural network trained to predict genotype dosage values. Stacks multiple hidden layers (256 → 128 → 64 neurons) with batch normalization, dropout regularization, and residual connections for stable training on high-dimensional genomic inputs. Outputs a soft dosage estimate over classes 0, 1, and 2.',
		strengths: ['Captures complex non-linear interactions between loci', 'Scales well as dataset size grows', 'Residual connections prevent vanishing gradients', 'GPU-accelerated via PyTorch (CUDA / Apple Metal)'],
		use_case: 'Best suited for large datasets where non-linear relationships between loci are expected. Outperforms simpler models when enough training examples are available to prevent overfitting.'
	},
	multi_log_regression: {
		label: 'Multinomial Logistic Regression',
		simple_description:
			'Our fastest and most transparent model. It examines each gene individually, assigns it a weight, and tallies up a score — straightforward enough that researchers can inspect exactly why it made each call.',
		simple_strengths: ['Completes in seconds on any dataset size', 'Easy to explain and audit', 'Reliable starting point for any new dataset'],
		simple_use_case: 'Best for quick results, smaller datasets, or situations where you need to clearly explain the model\'s decisions to others.',
		description:
			"A classical statistical classifier that models genotype probabilities using a linear combination of input features followed by a softmax transformation. Trains one set of weights per dosage class (0, 1, 2) using scikit-learn's multinomial solver. Highly interpretable — each feature weight directly quantifies its contribution to each class.",
		strengths: ['Fully interpretable coefficients', 'Extremely fast to train', 'Low memory footprint', 'Strong regularized baseline'],
		use_case: 'Ideal as a fast, transparent baseline or when the relationship between features and genotype dosage is approximately linear. Useful for debugging pipelines and establishing minimum performance expectations.'
	},
	hmm_dosage: {
		label: 'Hidden Markov Model',
		simple_description:
			'A model that understands that genes do not work in isolation — your DNA reads like a sentence, where each word gives context to the next. It follows your chromosome from start to finish, using nearby genes to sharpen every prediction.',
		simple_strengths: ['Naturally captures how genes influence their neighbours', 'Strong performance on sequential genomic data', 'Principled uncertainty estimates for each call'],
		simple_use_case: 'Best when the order and proximity of genes along the chromosome matter, or when working with phased genomic data.',
		description:
			'A probabilistic sequence model that treats each genomic site as an emission from an unobserved genetic state. Learns transition probabilities between hidden states along chromosomal position using the Baum-Welch (EM) algorithm. Captures linkage disequilibrium by modeling dependencies between adjacent loci in a principled, generative framework.',
		strengths: ['Explicitly captures sequential genomic structure', 'Probabilistic outputs support uncertainty quantification', 'Interpretable hidden states correspond to haplotype blocks', 'Efficient EM training converges reliably'],
		use_case: 'Most effective when linkage disequilibrium and chromosomal-order dependencies between loci are crucial. A natural fit for phased or partially phased genomic data.'
	},
	gnn_dosage: {
		label: 'Graph Neural Network',
		simple_description:
			'Our most relationship-aware model. It maps out your entire family tree and lets each person\'s known genetics inform predictions for their relatives — the same intuition a doctor uses when a condition "runs in the family," but at genomic scale.',
		simple_strengths: ['Uses family relationships to boost individual predictions', 'Handles multi-generational pedigrees', 'Improves most when relatives share genetic variants'],
		simple_use_case: 'Best when you have family or pedigree data and want the model to use those relationships to improve accuracy.',
		description:
			'A deep learning architecture that operates directly over pedigree graphs. Each individual is a node; edges represent parent-offspring or sibling relationships. Message-passing layers (256 → 128 → 64 hidden dimensions) aggregate genetic signals from neighbours, enabling the model to exploit Mendelian inheritance patterns across an entire family. GPU-accelerated via PyTorch.',
		strengths: ['Leverages pedigree structure for superior accuracy in family cohorts', 'Propagates genotype signals across multiple generations', 'Learns relational inheritance patterns end-to-end', 'Handles variable-size pedigrees through graph batching'],
		use_case: 'Most powerful when detailed pedigree data is available and individuals share genetic material with close relatives in the same dataset. Substantially outperforms non-relational models in deep pedigree cohorts.'
	},
	bayes_softmax3: {
		label: 'Bayesian Inference',
		simple_description:
			'Our most statistically rigorous model. It arrives at every genotype prediction by running thousands of simulations to build up a full picture of what the data supports — rather than making one quick calculation. The result is a prediction you can trust, especially on small or tricky datasets.',
		simple_strengths: ['More reliable predictions on small datasets', 'Does not overfit by jumping to conclusions', 'Incorporates existing biological knowledge as a starting point'],
		simple_use_case: 'Best when you need predictions you can trust even with limited data, or when the cost of a wrong call is high.',
		description:
			'A fully Bayesian multinomial logistic regression model fit via Markov Chain Monte Carlo (MCMC) using PyMC and the JAX/NUTS sampler. Places prior distributions over all model weights and generates a full posterior distribution rather than a single point estimate, enabling rigorous uncertainty quantification for each genotype call. Runs multiple parallel chains for convergence diagnostics.',
		strengths: ['Full posterior uncertainty quantification per prediction', 'Naturally incorporates allele-frequency priors', 'Convergence diagnostics (R-hat, ESS) via ArviZ', 'Robust to small sample sizes', 'GPU-accelerated sampling via JAX'],
		use_case: 'The strongest choice when calibrated confidence intervals on genotype calls are required, prior allele-frequency knowledge is available, or the dataset is small and overfitting is a concern. Slower to train but provides the most statistically rigorous outputs.'
	}
};

const MODEL_SIZE_INFO: Record<string, string> = {
	tiny: '250 individuals · 100 genomic sites · 5 generations. A minimal dataset for quick experiments and sanity checks.',
	small: '1,000 individuals · 1,000 genomic sites · 10 generations. A compact but realistic dataset suitable for most models.',
	medium: '5,000 individuals · 10,000 genomic sites · 25 generations. A full-scale dataset that exercises model capacity and captures richer genetic structure.',
	large: '25,000 individuals · 100,000 genomic sites · 50 generations. A large-scale dataset designed to push model limits and reflect real-world population sizes.'
};

const FALLBACK_INFO = {
	label: 'Custom Model',
	simple_description: 'This is a custom model. Refer to the model documentation for details on how it works.',
	simple_strengths: ['User-defined architecture'],
	simple_use_case: 'Application-specific. Consult the model configuration for details.',
	description: 'This model type does not have a built-in description. Refer to the model documentation for details.',
	strengths: ['User-defined architecture'],
	use_case: 'Application-specific. Consult the model configuration for details.'
};

export default function ModelDashboard({ model }: ModelDashboardProps) {
	const theme = useTheme();
	const isDark = theme.palette.mode === 'dark';
	const info = MODEL_TYPE_INFO[model.model_type] ?? FALLBACK_INFO;
	const [statsForNerds, setStatsForNerds] = useState(false);
	const sizeKey = model.model_name?.toLowerCase() ?? '';
	const sizeBlurb = MODEL_SIZE_INFO[sizeKey];

	return (
		<div style={{ marginTop: '1.25rem', padding: '1.5rem', backgroundColor: theme.palette.background.default, borderRadius: '8px' }}>
			<div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.25rem' }}>
				<h3 style={{ margin: 0 }}>Model Dashboard</h3>
				<FormControlLabel
					control={
						<Switch
							checked={statsForNerds}
							onChange={(e) => setStatsForNerds(e.target.checked)}
							size="small"
							sx={{
								'& .MuiSwitch-switchBase.Mui-checked': { color: '#452ee4' },
								'& .MuiSwitch-switchBase.Mui-checked + .MuiSwitch-track': { backgroundColor: '#452ee4' }
							}}
						/>
					}
					label="Stats for Nerds"
					labelPlacement="start"
					sx={{ margin: 0, gap: '0.4rem', '& .MuiFormControlLabel-label': { fontSize: '0.82rem', color: 'text.secondary' } }}
				/>
			</div>

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
						Model Size
					</p>
					<p style={{ margin: '0.4rem 0 0 0', fontSize: '1.1rem', fontWeight: 'bold' }}>
						{model.model_name ? model.model_name.charAt(0).toUpperCase() + model.model_name.slice(1) : model.model_name}
					</p>
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

			{/* Dataset size blurb */}
			{sizeBlurb && (
				<div style={{ marginBottom: '1.5rem' }}>
					<h4 style={{ marginTop: 0, marginBottom: '0.5rem' }}>Dataset Scale</h4>
					<p style={{ margin: 0, lineHeight: '1.6', opacity: 0.85 }}>{sizeBlurb}</p>
				</div>
			)}

			{/* Description */}
			<div style={{ marginBottom: '1.5rem' }}>
				<h4 style={{ marginTop: 0, marginBottom: '0.5rem' }}>About this Model</h4>
				<p style={{ margin: 0, lineHeight: '1.6', opacity: 0.85 }}>
					{statsForNerds ? info.description : info.simple_description}
				</p>
			</div>

			{/* Strengths */}
			<div style={{ marginBottom: '1.5rem' }}>
				<h4 style={{ marginTop: 0, marginBottom: '0.5rem' }}>Strengths</h4>
				<ul style={{ margin: 0, paddingLeft: '1.4rem', lineHeight: '1.8', opacity: 0.85 }}>
					{(statsForNerds ? info.strengths : info.simple_strengths).map((s) => (
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
					{statsForNerds ? 'Recommended Use Case' : 'Best For'}
				</p>
				<p style={{ margin: 0, lineHeight: '1.6' }}>{statsForNerds ? info.use_case : info.simple_use_case}</p>
			</div>
		</div>
	);
}
