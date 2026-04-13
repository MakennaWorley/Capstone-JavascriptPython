import { ThemeProvider, createTheme } from '@mui/material/styles';
import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import FamilyTreeVisualization from './FamilyTreeVisualization.js';

const theme = createTheme();

function renderWithTheme(ui: React.ReactElement) {
	return render(<ThemeProvider theme={theme}>{ui}</ThemeProvider>);
}

const basicData = {
	dataset: 'test_dataset',
	focus_id: 2,
	nodes: [
		{ id: 1, time: 1, observed: [0, 1] },
		{ id: 2, time: 0, observed: [1, null] },
		{ id: 3, time: 0, observed: [null, 0] }
	],
	edges: [
		{ source: 1, target: 2 },
		{ source: 1, target: 3 }
	]
};

const singleNodeData = {
	dataset: 'solo_dataset',
	focus_id: 5,
	nodes: [{ id: 5, time: 0, observed: [0] }],
	edges: []
};

describe('FamilyTreeVisualization', () => {
	it('renders the family tree heading with dataset and focus id', () => {
		renderWithTheme(<FamilyTreeVisualization data={basicData} />);
		expect(screen.getByRole('heading', { level: 3 })).toHaveTextContent(/test_dataset/i);
		expect(screen.getByRole('heading', { level: 3 })).toHaveTextContent(/Focus: 2/);
	});

	it('renders an SVG element', () => {
		renderWithTheme(<FamilyTreeVisualization data={basicData} />);
		expect(document.querySelector('svg')).toBeInTheDocument();
	});

	it('has an accessible aria-label on the SVG', () => {
		renderWithTheme(<FamilyTreeVisualization data={basicData} />);
		const svg = document.querySelector('svg');
		expect(svg).toHaveAttribute('aria-label', expect.stringContaining('test_dataset'));
	});

	it('renders with a single node and no edges without crashing', () => {
		renderWithTheme(<FamilyTreeVisualization data={singleNodeData} />);
		expect(screen.getByRole('heading', { level: 3 })).toHaveTextContent(/solo_dataset/i);
	});

	it('renders the title element inside the SVG', () => {
		renderWithTheme(<FamilyTreeVisualization data={basicData} />);
		const title = document.querySelector('svg title');
		expect(title?.textContent).toContain('test_dataset');
	});
});
