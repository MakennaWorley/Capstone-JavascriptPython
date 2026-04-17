import { ThemeProvider, createTheme } from '@mui/material/styles';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import Sidebar from './Sidebar.js';

const theme = createTheme();

function renderWithTheme(ui: React.ReactElement) {
	return render(<ThemeProvider theme={theme}>{ui}</ThemeProvider>);
}

const defaultProps = {
	darkMode: false,
	onThemeToggle: vi.fn(),
	debugMode: false,
	onDebugToggle: vi.fn(),
	onCreateDataset: vi.fn()
};

describe('Sidebar', () => {
	it('renders the Create Dataset button', () => {
		renderWithTheme(<Sidebar {...defaultProps} />);
		expect(screen.getByRole('button', { name: /Create new dataset/i })).toBeInTheDocument();
	});

	it('renders the navigation landmark', () => {
		renderWithTheme(<Sidebar {...defaultProps} />);
		expect(screen.getByRole('navigation', { name: /Main navigation/i })).toBeInTheDocument();
	});

	it('calls onCreateDataset when Create Dataset button is clicked', async () => {
		const onCreateDataset = vi.fn();
		renderWithTheme(<Sidebar {...defaultProps} onCreateDataset={onCreateDataset} />);
		const user = userEvent.setup();
		await user.click(screen.getByRole('button', { name: /Create new dataset/i }));
		expect(onCreateDataset).toHaveBeenCalledTimes(1);
	});

	it('calls onThemeToggle when the theme button is clicked', async () => {
		const onThemeToggle = vi.fn();
		renderWithTheme(<Sidebar {...defaultProps} onThemeToggle={onThemeToggle} />);
		const user = userEvent.setup();
		await user.click(screen.getByRole('button', { name: /Switch to Dark Mode/i }));
		expect(onThemeToggle).toHaveBeenCalledTimes(1);
	});

	it('shows Switch to Light Mode label when darkMode is true', () => {
		renderWithTheme(<Sidebar {...defaultProps} darkMode={true} />);
		expect(screen.getByRole('button', { name: /Switch to Light Mode/i })).toBeInTheDocument();
	});

	it('calls onDebugToggle when the debug button is clicked', async () => {
		const onDebugToggle = vi.fn();
		renderWithTheme(<Sidebar {...defaultProps} onDebugToggle={onDebugToggle} />);
		const user = userEvent.setup();
		await user.click(screen.getByRole('button', { name: /Debug: OFF/i }));
		expect(onDebugToggle).toHaveBeenCalledTimes(1);
	});

	it('shows Debug: ON label when debugMode is true', () => {
		renderWithTheme(<Sidebar {...defaultProps} debugMode={true} />);
		expect(screen.getByRole('button', { name: /Debug: ON/i })).toBeInTheDocument();
	});

	it('opens the drawer when the menu toggle button is clicked', async () => {
		renderWithTheme(<Sidebar {...defaultProps} />);
		const user = userEvent.setup();
		await user.click(screen.getByRole('button', { name: /Open navigation/i }));
		expect(screen.getByRole('button', { name: /Close navigation/i })).toBeInTheDocument();
	});
});
