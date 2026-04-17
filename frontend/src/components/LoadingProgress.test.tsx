import { ThemeProvider, createTheme } from '@mui/material/styles';
import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import LoadingProgress from './LoadingProgress.js';

const theme = createTheme();

function renderWithTheme(ui: React.ReactElement) {
	return render(<ThemeProvider theme={theme}>{ui}</ThemeProvider>);
}

describe('LoadingProgress', () => {
	it('renders nothing when isLoading is false', () => {
		const { container } = renderWithTheme(<LoadingProgress isLoading={false} message="Loading…" />);
		expect(container.firstChild).toBeNull();
	});

	it('renders the status container when isLoading is true', () => {
		renderWithTheme(<LoadingProgress isLoading={true} message="Loading data…" />);
		expect(screen.getByRole('status')).toBeInTheDocument();
	});

	it('displays the provided message text', () => {
		renderWithTheme(<LoadingProgress isLoading={true} message="Training model…" />);
		expect(screen.getByText('Training model…')).toBeInTheDocument();
	});

	it('has aria-live="polite" for accessibility', () => {
		renderWithTheme(<LoadingProgress isLoading={true} message="Loading…" />);
		expect(screen.getByRole('status')).toHaveAttribute('aria-live', 'polite');
	});
});
