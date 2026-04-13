import { type Theme, createTheme } from '@mui/material/styles';

export function createAppTheme(mode: 'light' | 'dark'): Theme {
	return createTheme({
		palette: {
			mode,
			primary: {
				main: '#452ee4',
				dark: '#241291',
				contrastText: '#ffffff'
			},
			background: mode === 'dark' ? { default: '#000000', paper: '#010101' } : { default: '#ffffff', paper: '#f5f5f5' },
			divider: mode === 'dark' ? '#333333' : '#cccccc'
		},
		typography: {
			fontFamily: 'Arial, sans-serif'
		},
		components: {
			MuiButton: {
				styleOverrides: {
					root: {
						textTransform: 'none'
					}
				}
			}
		}
	});
}
