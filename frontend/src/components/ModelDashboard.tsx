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
			"A Deep Neural Network (DNN) is modelled loosely after the human brain — it is made up of layers of interconnected nodes that each learn to recognise a small piece of a pattern. Stack enough of these layers together and the network can pick up on extraordinarily subtle signals in the data that no human analyst would ever spot manually. For this project, the DNN looks at the genetic data of a person's relatives and learns, through thousands of examples, what combination of signals tends to predict each possible genotype. It does not need to be told which features matter — it figures that out on its own. The trade-off is that it needs a reasonably large amount of training data to do this well, and it can take longer to train than simpler models. But when conditions are right, it is our most powerful tool.",
		simple_strengths: [
			'Discovers complex patterns in genetic data automatically, without needing to be told what to look for',
			'Becomes significantly more accurate as the training dataset grows larger',
			'Handles many thousands of genetic sites at once without slowing down',
			'Uses techniques like batch normalisation and dropout to stay reliable and avoid memorising noise',
			'Can run on a GPU for dramatically faster training on large datasets'
		],
		simple_use_case:
			'The Deep Neural Network shines when you have a lot of data to work with. If your dataset includes hundreds or thousands of individuals and thousands of genomic sites, this model has enough examples to learn genuinely useful patterns rather than just memorising quirks in the training set. It is the right choice when you want the most accurate predictions possible and can afford to wait a little longer for training to complete — or when you have a GPU available to speed things up. It is less well suited to very small datasets, where simpler models like logistic regression or Bayesian inference tend to be more reliable.',
		description:
			'A fully connected feedforward neural network (PyTorch) trained to predict per-site allele dosage (0, 1, 2) for masked individuals. The architecture stacks three hidden layers (256 → 128 → 64 neurons), each followed by batch normalisation and ReLU activation. Residual (skip) connections are added between compatible layer widths to mitigate vanishing gradients. Dropout (p=0.3) is applied during training for regularisation. The output layer uses softmax over three classes. Input features are constructed via k-hop relative aggregation over the pedigree graph: for each masked individual at each genomic site, the feature vector is [mean_dosage_of_k_hop_relatives, fraction_of_relatives_observed, count_of_relatives]. Training uses cross-entropy loss with class weights inversely proportional to class frequency to handle dosage imbalance. Optimised with Adam; early stopping monitors validation loss with patience of 10 epochs. Supports CUDA and Apple Metal (MPS) acceleration.',
		strengths: [
			'Captures non-linear, high-order interactions between loci that linear models cannot represent',
			'Residual connections stabilise gradient flow in deep stacks and accelerate convergence',
			'Batch normalisation reduces internal covariate shift, enabling higher learning rates',
			'Class-weighted cross-entropy handles dosage class imbalance without resampling',
			'Early stopping prevents overfitting when validation loss plateaus',
			'Scales efficiently to high-dimensional inputs (100k+ genomic sites) on GPU',
			'PyTorch backend supports CUDA (NVIDIA) and MPS (Apple Silicon) acceleration'
		],
		use_case:
			'Best suited for medium to large datasets (1,000+ individuals, 1,000+ sites) where non-linear feature interactions are expected and sufficient training data exists to prevent overfitting. Outperforms logistic regression when pedigree-derived features have complex dosage dependencies. Less suitable than Bayesian inference for small datasets or when posterior uncertainty estimates are required. Slower to train than logistic regression but typically achieves the highest raw accuracy on large cohorts. Recommended when GPU hardware is available.'
	},
	multi_log_regression: {
		label: 'Multinomial Logistic Regression',
		simple_description:
			"Logistic regression is one of the oldest and most well-understood tools in statistics. At its core, it works by assigning a numerical weight to each input feature — in this case, things like how many of a person's relatives carry a particular allele — and then adding those weighted signals together to produce a score for each possible genotype. The genotype with the highest score wins. Because every decision traces back to a simple sum of weights, researchers can open up the model and read exactly what it learned. There are no hidden layers, no mysterious representations — just interpretable numbers. It is the baseline every other model has to beat, and it often performs surprisingly well given how simple it is.",
		simple_strengths: [
			'Fully transparent — you can inspect exactly why it made each prediction',
			'Trains in seconds even on the largest datasets',
			'Rarely overfits, making it reliable even with limited data',
			'A strong and honest baseline: if a fancier model cannot beat it, something is wrong',
			'Works well when the relationship between features and genotype is roughly linear'
		],
		simple_use_case:
			'Logistic regression is the right starting point for almost any analysis. If you are new to a dataset and want a quick, honest read of how predictable the genotypes are, start here. It is also the best choice when you need to explain your results to someone without a machine-learning background — every weight in the model has a direct, interpretable meaning. On smaller datasets it often matches or beats more complex models because it does not have enough parameters to overfit. The one scenario where it tends to fall short is when the patterns in the data are genuinely non-linear and complex, which is when you would consider upgrading to a DNN or GNN.',
		description:
			"A multinomial logistic regression classifier (scikit-learn) that models P(dosage | features) via softmax over three output classes (0, 1, 2). Input features are the same k-hop pedigree aggregation vectors used by other models: [mean_dosage_of_relatives, fraction_observed, count_relatives] per genomic site per masked individual. The model learns one weight vector per class, solved via the L-BFGS multinomial solver with L2 regularisation (C=1.0). Class weights are set to 'balanced' to compensate for dosage distribution skew. Because the decision boundary is a hyperplane in feature space, each coefficient has a direct interpretation: a positive weight on mean_dosage_of_relatives for dosage class 2 means that higher average relative dosage increases the log-odds of predicting class 2. This is the only model in the benchmark that provides this level of interpretability without post-hoc analysis.",
		strengths: [
			'Coefficients are directly interpretable: each weight quantifies the contribution of one feature to one dosage class',
			'L-BFGS solver converges in seconds on any dataset in this benchmark',
			'L2 regularisation controls overfitting without requiring hyperparameter search',
			'Balanced class weights handle dosage imbalance without resampling or custom loss functions',
			'No GPU required; runs on any hardware with minimal memory overhead',
			'Serves as the primary baseline for assessing whether complexity in other models is justified'
		],
		use_case:
			'Use as the first model on any new dataset to establish a performance floor. If logistic regression achieves high accuracy, the genetic signal is linearly separable and more complex models may not add meaningful value. If accuracy is poor, non-linear structure is present and a DNN or GNN is warranted. Always compare every other model against this baseline before drawing conclusions about model superiority. Also the recommended model when auditability of predictions is a hard requirement.'
	},
	hmm_dosage: {
		label: 'Hidden Markov Model',
		simple_description:
			"A Hidden Markov Model (HMM) is built on a simple but powerful idea: the current state of something depends on what came just before it. In genetics, this maps beautifully onto the structure of a chromosome — nearby sites along your DNA are not independent, they influence each other because of how inheritance works (a phenomenon called linkage disequilibrium). The HMM reads a person's chromosome like a story, moving from site to site and using what it just saw to inform what it expects next. Underneath, it assumes there are a small number of hidden genetic states the data could be in at any point, and it learns to recognise which state fits best given the observed pattern. It does not need a family tree — it finds structure within the sequence itself.",
		simple_strengths: [
			'Naturally understands that nearby genes on a chromosome influence each other',
			'Reads genetic data as a sequence rather than treating each site independently',
			'Produces a probability for each possible genotype, not just a single guess',
			'Well-suited to data where the order of measurements carries meaning',
			'Trains efficiently without requiring a GPU'
		],
		simple_use_case:
			'The Hidden Markov Model is at its best when the structure of the genome itself carries the signal — specifically, when genes that sit close together on the same chromosome tend to be inherited together. This is very common in real genetic data. If your dataset preserves the chromosomal order of sites (rather than treating them as an unordered bag of measurements), the HMM can exploit that structure in a way no other model here can. It is also a strong choice when your dataset is moderate in size and you want well-calibrated confidence scores alongside each prediction. If you are unsure whether chromosomal order matters for your data, running the HMM alongside logistic regression is a good way to find out.',
		description:
			'A Gaussian Hidden Markov Model (hmmlearn) trained to infer per-site allele dosage by treating each individual as an observed sequence across genomic sites. The model assumes K=3 hidden states (corresponding loosely to dosage classes 0, 1, 2) with Gaussian emission distributions and a learned transition matrix. State parameters are estimated via Baum-Welch (EM). Emission means and covariances are initialised from empirical statistics of labelled examples (semi-supervised init) to accelerate convergence and avoid degenerate solutions. At inference time, the Viterbi algorithm decodes the most probable hidden state sequence for each individual, and posterior state probabilities are used to produce soft dosage class probabilities. The sequential structure of the model captures linkage disequilibrium — correlation between nearby sites — that i.i.d. models ignore entirely.',
		strengths: [
			'Explicitly models chromosomal-position dependencies via the learned transition matrix',
			'Semi-supervised initialisation from label statistics prevents degenerate local optima in EM',
			'Viterbi decoding produces a globally consistent state sequence rather than per-site greedy decisions',
			'Posterior state probabilities provide well-calibrated soft class estimates',
			'Interpretable: hidden state means and transition probabilities can be inspected directly',
			'Efficient EM convergence with no GPU required',
			'Naturally handles variable-length sequences without padding'
		],
		use_case:
			'Most effective when genomic sites in the feature matrix are ordered by chromosomal position and linkage disequilibrium between adjacent sites is non-negligible. Performance degrades if sites are shuffled or drawn from unrelated chromosomal regions. Recommended when phasing information is available or where sequential correlation structure is a known biological feature. Not recommended when the feature set mixes sites from multiple chromosomes without positional ordering.'
	},
	gnn_dosage: {
		label: 'Graph Neural Network',
		simple_description:
			"A Graph Neural Network (GNN) is designed from the ground up to reason about relationships. Instead of treating each individual as an isolated data point, it builds a map of the entire family — who is whose parent, who are siblings, how everyone connects. It then passes messages along the edges of that map: each person shares their genetic information with their relatives, and those relatives share back, so that by the end every prediction is informed by the whole connected family rather than just one person's own data. This mirrors how inheritance actually works — if a grandparent carries a particular variant, there is a meaningful chance their grandchildren do too, and the GNN learns to exploit exactly that kind of multi-generational signal. It is our most family-aware model.",
		simple_strengths: [
			'Treats the family tree as the core input, not just an afterthought',
			'Passes genetic signals up and down through multiple generations simultaneously',
			'Predictions for one person are informed by their parents, children, and more distant relatives',
			'Learns inheritance patterns end-to-end from the data itself',
			'Handles families of any size or shape, from small pedigrees to large extended clans'
		],
		simple_use_case:
			'The Graph Neural Network is the right choice whenever the family structure of your data is as informative as the genetic measurements themselves — which, in ancestral inference, it almost always is. If your dataset includes parents, grandparents, or other close relatives of the individuals you are trying to predict, the GNN will use those connections to make substantially better predictions than any model that ignores them. It performs best on medium to large datasets with deep, connected pedigrees. For very small families or isolated individuals with no known relatives in the dataset, its advantage over simpler models will be smaller. It also benefits from GPU acceleration on larger datasets.',
		description:
			'A graph convolutional network (PyTorch Geometric) that constructs a feature-correlation graph from the input data and applies message-passing convolutions to aggregate genetic signals across connected individuals. Each individual is a node; edges are drawn between individuals whose feature vectors exceed a Pearson correlation threshold (default 0.5), encoding genetic similarity independently of pedigree topology. Node features are the same k-hop pedigree aggregation vectors used by other models. The architecture stacks three GraphConv layers (256 → 128 → 64 hidden dimensions) with ReLU activations, followed by global mean pooling to produce a graph-level embedding passed through a linear classifier. Trained with cross-entropy loss and Adam. Supports mini-batch training via PyTorch Geometric DataLoader for memory-efficient handling of large pedigrees. GPU-accelerated.',
		strengths: [
			'Constructs edges from feature correlation, capturing genetic similarity that the pedigree graph alone may not encode',
			'Message-passing aggregates information from all graph neighbours, not just direct parents/children',
			'Global mean pooling produces size-invariant graph embeddings that generalise across pedigree shapes',
			'Mini-batch DataLoader enables training on pedigrees too large to fit in memory at once',
			'GPU-accelerated via PyTorch Geometric sparse kernels',
			'Learns relational inheritance patterns end-to-end without hand-crafted pedigree features',
			'Outperforms non-relational models on datasets with dense, multi-generational family structure'
		],
		use_case:
			'Most powerful on datasets with deep pedigrees (10+ generations) and high connectivity, where many individuals share recent common ancestors. The correlation-based edge construction means the model can also exploit genetic similarity between individuals not directly related in the recorded pedigree. Requires more memory and training time than logistic regression or the HMM; GPU acceleration strongly recommended for medium and large datasets. Performance advantage over the DNN diminishes when pedigree connectivity is sparse.'
	},
	bayes_softmax3: {
		label: 'Bayesian Inference',
		simple_description:
			'Bayesian inference is a fundamentally different philosophy of prediction. Most models give you a single answer: "this person has genotype 1." Bayesian models give you a full picture: "there is a 70% chance it is 1, a 25% chance it is 0, and a 5% chance it is 2 — and here is how confident we are in those numbers." It achieves this by running thousands of simulated sampling steps (called Markov Chain Monte Carlo, or MCMC) to explore all the ways the data could be explained, then combining them into a final probability distribution. This means it is naturally cautious — it will not overcommit to an answer when the data is ambiguous. It also starts with a prior belief about how common each genotype is across the population, which helps it make sensible predictions even when very little data is available.',
		simple_strengths: [
			'Gives a full probability distribution for each prediction, not just a single answer',
			'Naturally cautious — uncertainty in the data is reflected in uncertain predictions',
			'Performs well even when the training dataset is very small',
			'Can incorporate existing knowledge about allele frequencies as a starting point',
			'The most trustworthy model when the consequences of a wrong prediction are serious'
		],
		simple_use_case:
			'Bayesian inference is the right choice when trust matters as much as accuracy. Because it produces a full probability distribution rather than a single answer, you always know how confident the model is — and when it is uncertain, it says so rather than bluffing. This makes it particularly valuable when the dataset is small and other models might overfit, or when a wrong prediction has real consequences and you need to know the risk. It is also the best option if you have prior scientific knowledge about allele frequencies that you want to incorporate into the analysis. The trade-off is speed: because it runs thousands of sampling steps, it is the slowest model to train, especially on large datasets. But for the right problem, the extra rigour is worth it.',
		description:
			'A fully Bayesian multinomial logistic regression model implemented in PyMC and sampled via the No-U-Turn Sampler (NUTS) with JAX as the computational backend. Rather than optimising a single point estimate of model weights, NUTS draws samples from the full joint posterior P(weights | data) using Hamiltonian Monte Carlo dynamics. Priors are placed over all weight matrices: Normal(0, 1) for feature weights and Normal(0, 0.5) for per-generation group-level intercepts (hierarchical structure). The likelihood is a categorical distribution parameterised by softmax over three dosage classes. By default the model runs 4 parallel chains with 1,000 tuning steps and 500 draw steps each, producing 2,000 posterior samples. At inference time, predictions are made by averaging the softmax outputs across all posterior samples, yielding calibrated class probabilities. Convergence is assessed via R-hat (target < 1.01) and effective sample size (ESS) diagnostics computed by ArviZ.',
		strengths: [
			'Produces a full posterior distribution over model weights — every prediction comes with a credible interval',
			'NUTS sampler is geometrically ergodic — guaranteed to converge to the true posterior given sufficient samples',
			'Hierarchical priors over generation-level intercepts regularise predictions for generations with few observations',
			'R-hat and ESS diagnostics provide objective convergence evidence unavailable in frequentist models',
			'Posterior predictive averaging produces better-calibrated class probabilities than a single MAP estimate',
			'Robust to small sample sizes due to prior regularisation',
			'JAX backend enables GPU/TPU-accelerated HMC with just-in-time compilation'
		],
		use_case:
			'The strongest choice when statistically rigorous uncertainty quantification is required: credible intervals on dosage probabilities, posterior predictive checks, or sensitivity analysis over prior choices. Hierarchical structure makes it particularly well-suited to datasets spanning multiple generations with unequal sample sizes per generation. Slower than all other models by a significant margin — expect minutes to hours on medium/large datasets without GPU acceleration. Not recommended when training speed is a hard constraint or when the dataset is large enough that the DNN or GNN provides equivalent accuracy without the sampling overhead.'
	}
};

const MODEL_SIZE_INFO: Record<
	string,
	{
		simple: string;
		nerd: string;
	}
> = {
	tiny: {
		simple: 'This model was trained on a small simulated family — 250 individuals spread across 5 generations, with 100 genetic sites measured per person. Think of it as a proof-of-concept: small enough to train in seconds, useful for quickly checking whether the model is working as expected. Because the dataset is so compact, the model has seen fewer examples and may not generalize as well to larger, more complex families.',
		nerd: '250 diploid individuals · 100 genomic sites · 5 generations · 50 samples per generation. Minimal-scale dataset intended for pipeline validation and rapid iteration. Limited feature diversity; expect higher variance in model performance. Useful for debugging and sanity checks but not representative of real-world cohort complexity.'
	},
	small: {
		simple: 'This model was trained on a moderate-sized simulated family — 1,000 individuals across 10 generations, with 1,000 genetic sites per person. This is a solid starting point: the model has seen enough variation to learn meaningful patterns, but training still completes quickly. It strikes a good balance between speed and reliability for most experiments.',
		nerd: '1,000 diploid individuals · 1,000 genomic sites · 10 generations · 50 samples per generation. Compact but representative dataset. Sufficient feature diversity for most model architectures to learn stable decision boundaries. Recommended for exploratory comparisons between model types before scaling up.'
	},
	medium: {
		simple: 'This model was trained on a large simulated family — 5,000 individuals across 25 generations, with 10,000 genetic sites per person. At this scale, the model has been exposed to a wide range of family structures and inheritance patterns. Training takes longer, but the result is a model that can handle more complex scenarios and is generally more accurate.',
		nerd: '5,000 diploid individuals · 10,000 genomic sites · 25 generations · 100 samples per generation. Full-scale dataset that exercises model capacity meaningfully. Captures richer linkage structure and deeper pedigree relationships. Appropriate for benchmarking all five architectures under realistic conditions.'
	},
	large: {
		simple: 'This model was trained on a very large simulated family — 25,000 individuals across 50 generations, with 100,000 genetic sites per person. This is the most demanding scale in the benchmark, designed to reflect the complexity of real-world population studies. Models trained here have seen an enormous range of genetic variation and are the most thoroughly tested — though they also take the longest to run.',
		nerd: '25,000 diploid individuals · 100,000 genomic sites · 50 generations · 500 samples per generation. Large-scale dataset designed to stress-test model capacity and expose performance degradation under high-dimensional inputs. Reflects realistic population cohort complexity. GPU acceleration strongly recommended for DNN, GNN, and Bayesian models at this scale.'
	}
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
		<div style={{ marginTop: '1.25rem' }}>
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
					<h3 style={{ marginTop: 0, marginBottom: '0.5rem' }}>Training Dataset</h3>
					<p style={{ margin: 0, lineHeight: '1.6', opacity: 0.85 }}>{sizeBlurb.simple}</p>
					{statsForNerds && (
						<p style={{ margin: '0.75rem 0 0', lineHeight: '1.6', opacity: 0.85, fontFamily: 'monospace', fontSize: '0.9rem' }}>
							{sizeBlurb.nerd}
						</p>
					)}
					<p style={{ margin: '0.75rem 0 0', lineHeight: '1.6', opacity: 0.85 }}>
						All data is fully simulated — every individual is a computer-generated person, and their DNA is produced by a genetic
						simulator called <strong>msprime</strong> that mimics how real inheritance works. Because the simulation controls everything,
						the true genotype of every person is always known, even for the ones intentionally left out. This is what lets us measure
						exactly how well the model performed.
					</p>
				</div>
			)}

			{/* Training pipeline */}
			<div style={{ marginBottom: '1.5rem' }}>
				<h3 style={{ marginTop: 0, marginBottom: '0.5rem' }}>{statsForNerds ? 'Training Pipeline' : 'How the Model Was Trained'}</h3>
				{statsForNerds ? (
					<div style={{ lineHeight: '1.6', opacity: 0.85 }}>
						<p style={{ margin: '0 0 0.75rem' }}>
							Training follows a 3-phase pipeline applied to three dataset splits derived from the simulation output.
						</p>
						<p style={{ margin: '0 0 0.5rem' }}>
							<strong>Phase 1 — Initial Training:</strong> The model is fit on the training split using the full labelled feature
							matrix. For each masked individual at each genomic site, the input feature vector is{' '}
							<code>[mean_dosage_of_k_hop_relatives, fraction_of_relatives_observed, count_of_relatives]</code>, constructed via k-hop
							relative aggregation over the pedigree graph. The target is allele dosage (0, 1, or 2).
						</p>
						<p style={{ margin: '0 0 0.5rem' }}>
							<strong>Phase 2 — Cross-Validation &amp; Retraining:</strong> The trained model is evaluated on the validation split using
							5-fold cross-validation (KFold, shuffle=True, random_state=123). Per-fold metrics are averaged to produce a robust
							estimate of generalisation performance. The model is then retrained from scratch on the combined train + validation data
							to maximise the information available before final testing.
						</p>
						<p style={{ margin: '0 0 0.75rem' }}>
							<strong>Phase 3 — Final Evaluation:</strong> The retrained model is applied once to a held-out test split that was never
							seen during training or cross-validation. Metrics reported here (precision, recall, F1, ROC/PR AUC, confusion matrix) are
							all computed from this final phase.
						</p>
						<p style={{ margin: 0 }}>
							Results are cached per (model_name, model_type, test_dataset) tuple and served from the log on subsequent requests to
							avoid redundant recomputation.
						</p>
					</div>
				) : (
					<p style={{ margin: 0, lineHeight: '1.6', opacity: 0.85 }}>
						Before training, a portion of individuals in the dataset were hidden — their genotypes were removed to simulate real-world
						missing ancestors. The model was then shown only the remaining individuals and asked to learn the relationship between a
						person's relatives' DNA and their own. Training happened in stages: first on a core set of examples, then validated and
						refined, and finally tested on a completely separate group of hidden individuals the model had never encountered. This staged
						approach helps ensure the model genuinely learned to infer genetics — not just memorise the training data.
					</p>
				)}
			</div>

			{/* Description */}
			<div style={{ marginBottom: '1.5rem' }}>
				<h3 style={{ marginTop: 0, marginBottom: '0.5rem' }}>About this Model</h3>
				<p style={{ margin: 0, lineHeight: '1.6', opacity: 0.85 }}>{statsForNerds ? info.description : info.simple_description}</p>
			</div>

			{/* Strengths */}
			<div style={{ marginBottom: '1.5rem' }}>
				<h3 style={{ marginTop: 0, marginBottom: '0.5rem' }}>Strengths</h3>
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
