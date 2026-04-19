import AddIcon from '@mui/icons-material/Add';
import BugReportIcon from '@mui/icons-material/BugReport';
import CloseIcon from '@mui/icons-material/Close';
import DarkModeIcon from '@mui/icons-material/DarkMode';
import LightModeIcon from '@mui/icons-material/LightMode';
import MenuIcon from '@mui/icons-material/Menu';
import SchoolIcon from '@mui/icons-material/School';
import { Box, Divider, Drawer, IconButton, List, ListItemButton, ListItemIcon } from '@mui/material';
import { useState } from 'react';

const DRAWER_WIDTH_OPEN = 280;
const DRAWER_WIDTH_CLOSED = 80;

interface SidebarProps {
	darkMode: boolean;
	onThemeToggle: () => void;
	debugMode: boolean;
	onDebugToggle: () => void;
	nerdMode: boolean;
	onNerdModeChange: (value: boolean) => void;
	onCreateDataset: () => void;
}

export default function Sidebar({ darkMode, onThemeToggle, debugMode, onDebugToggle, nerdMode, onNerdModeChange, onCreateDataset }: SidebarProps) {
	const [open, setOpen] = useState(false);

	const drawerWidth = open ? DRAWER_WIDTH_OPEN : DRAWER_WIDTH_CLOSED;

	return (
		<Drawer
			variant="permanent"
			className="sidebar-drawer"
			sx={{
				width: drawerWidth,
				flexShrink: 0,
				transition: 'width 0.3s ease',
				'& .MuiDrawer-paper': {
					width: drawerWidth
				}
			}}
		>
			<Box component="nav" aria-label="Main navigation" sx={{ overflow: 'auto', display: 'flex', flexDirection: 'column', flex: 1 }}>
				{/* Toggle Button at top of sidebar */}
				<Box sx={{ display: 'flex', justifyContent: open ? 'flex-end' : 'center', pt: 1, px: open ? 1 : 0 }}>
					<IconButton
						onClick={() => setOpen(!open)}
						aria-label={open ? 'Close navigation' : 'Open navigation'}
						className="sidebar-toggle-internal"
					>
						{open ? <CloseIcon /> : <MenuIcon />}
					</IconButton>
				</Box>

				<List sx={{ px: open ? 1 : 0 }}>
					<ListItemButton
						onClick={onCreateDataset}
						title="Create new dataset"
						aria-label="Create new dataset"
						className="sidebar-btn-create"
						sx={{
							margin: open ? '0.5rem' : '0.5rem auto',
							width: open ? 'auto' : '56px',
							justifyContent: open ? 'flex-start' : 'center',
							px: open ? 1 : 0
						}}
					>
						<ListItemIcon sx={{ color: 'white', minWidth: open ? '40px' : 'auto' }}>
							<AddIcon />
						</ListItemIcon>
						{open && <span className="sidebar-item-label">Create Dataset</span>}
					</ListItemButton>
				</List>

				<Divider sx={{ my: 1, backgroundColor: 'var(--color-primary)', mx: open ? 1 : 0.5 }} />

				{/* Bottom menu items */}
				<Box sx={{ marginTop: 'auto', pb: 2, px: open ? 1 : 0 }}>
					<List>
						<ListItemButton
							onClick={onThemeToggle}
							title={darkMode ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
							aria-label={darkMode ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
							className={darkMode ? 'sidebar-btn-theme-dark' : 'sidebar-btn-theme-light'}
							sx={{
								margin: open ? '0.5rem' : '0.5rem auto',
								width: open ? 'auto' : '56px',
								justifyContent: open ? 'flex-start' : 'center',
								px: open ? 1 : 0
							}}
						>
							<ListItemIcon sx={{ color: darkMode ? 'white' : '#111', minWidth: open ? '40px' : 'auto' }}>
								{darkMode ? <LightModeIcon /> : <DarkModeIcon />}
							</ListItemIcon>
							{open && <span className="sidebar-item-label">{darkMode ? 'Light Mode' : 'Dark Mode'}</span>}
						</ListItemButton>
						<ListItemButton
							onClick={() => onNerdModeChange(!nerdMode)}
							title={`Stats for Nerds: ${nerdMode ? 'ON' : 'OFF'}`}
							aria-label={`Stats for Nerds: ${nerdMode ? 'ON' : 'OFF'}`}
							className={nerdMode ? 'sidebar-btn-nerd-on' : 'sidebar-btn-nerd-off'}
							sx={{
								margin: open ? '0.5rem' : '0.5rem auto',
								width: open ? 'auto' : '56px',
								justifyContent: open ? 'flex-start' : 'center',
								px: open ? 1 : 0
							}}
						>
							<ListItemIcon sx={{ color: 'white', minWidth: open ? '40px' : 'auto' }}>
								<SchoolIcon />
							</ListItemIcon>
							{open && <span className="sidebar-item-label">{nerdMode ? 'Stats for Nerds: ON' : 'Stats for Nerds: OFF'}</span>}
						</ListItemButton>
						<ListItemButton
							onClick={onDebugToggle}
							title={`Debug: ${debugMode ? 'ON' : 'OFF'}`}
							aria-label={`Debug: ${debugMode ? 'ON' : 'OFF'}`}
							className={debugMode ? 'sidebar-btn-debug-on' : 'sidebar-btn-debug-off'}
							sx={{
								margin: open ? '0.5rem' : '0.5rem auto',
								width: open ? 'auto' : '56px',
								justifyContent: open ? 'flex-start' : 'center',
								px: open ? 1 : 0
							}}
						>
							<ListItemIcon sx={{ color: 'white', minWidth: open ? '40px' : 'auto' }}>
								<BugReportIcon />
							</ListItemIcon>
							{open && <span className="sidebar-item-label">{debugMode ? 'Debug: ON' : 'Debug: OFF'}</span>}
						</ListItemButton>
					</List>
				</Box>
			</Box>
		</Drawer>
	);
}
