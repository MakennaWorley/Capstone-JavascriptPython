import AddIcon from '@mui/icons-material/Add';
import BugReportIcon from '@mui/icons-material/BugReport';
import CloseIcon from '@mui/icons-material/Close';
import MenuIcon from '@mui/icons-material/Menu';
import { Box, Divider, Drawer, IconButton, List, ListItemButton, ListItemIcon } from '@mui/material';
import { useState } from 'react';

const DRAWER_WIDTH_OPEN = 280;
const DRAWER_WIDTH_CLOSED = 80;

interface SidebarProps {
	debugMode: boolean;
	onDebugToggle: () => void;
	onCreateDataset: () => void;
}

export default function Sidebar({ debugMode, onDebugToggle, onCreateDataset }: SidebarProps) {
	const [open, setOpen] = useState(true);

	const drawerWidth = open ? DRAWER_WIDTH_OPEN : DRAWER_WIDTH_CLOSED;

	return (
		<>
			<Drawer
				variant="permanent"
				sx={{
					width: drawerWidth,
					flexShrink: 0,
					transition: 'width 0.3s ease',
					'& .MuiDrawer-paper': {
						width: drawerWidth,
						boxSizing: 'border-box',
						backgroundColor: '#1a1a1a',
						borderRight: '1px solid #646cff',
						transition: 'width 0.3s ease',
						overflowX: 'hidden'
					}
				}}
			>
				<Box sx={{ overflow: 'auto', display: 'flex', flexDirection: 'column', flex: 1 }}>
					<List sx={{ px: open ? 1 : 0, mt: '3.5rem' }}>
						<ListItemButton
							onClick={onCreateDataset}
							title="Create new dataset"
							sx={{
								margin: open ? '0.5rem' : '0.5rem auto',
								width: open ? 'auto' : '56px',
								borderRadius: '8px',
								backgroundColor: '#646cff',
								color: 'white',
								justifyContent: open ? 'flex-start' : 'center',
								px: open ? 1 : 0,
								'&:hover': {
									backgroundColor: '#747eff'
								}
							}}
						>
							<ListItemIcon sx={{ color: 'white', minWidth: open ? '40px' : 'auto' }}>
								<AddIcon />
							</ListItemIcon>
							{open && <span style={{ marginLeft: '0.5rem', whiteSpace: 'nowrap' }}>Create Dataset</span>}
						</ListItemButton>
					</List>

					<Divider sx={{ my: 1, backgroundColor: '#646cff', mx: open ? 1 : 0.5 }} />

					{/* Bottom menu items */}
					<Box sx={{ marginTop: 'auto', pb: 2, px: open ? 1 : 0 }}>
						<List>
							<ListItemButton
								onClick={onDebugToggle}
								title={`Debug: ${debugMode ? 'ON' : 'OFF'}`}
								sx={{
									margin: open ? '0.5rem' : '0.5rem auto',
									width: open ? 'auto' : '56px',
									borderRadius: '8px',
									backgroundColor: debugMode ? '#4caf50' : '#555',
									color: 'white',
									justifyContent: open ? 'flex-start' : 'center',
									px: open ? 1 : 0,
									'&:hover': {
										backgroundColor: debugMode ? '#45a049' : '#666'
									}
								}}
							>
								<ListItemIcon sx={{ color: 'white', minWidth: open ? '40px' : 'auto' }}>
									<BugReportIcon />
								</ListItemIcon>
								{open && <span style={{ marginLeft: '0.5rem', whiteSpace: 'nowrap' }}>{debugMode ? 'Debug: ON' : 'Debug: OFF'}</span>}
							</ListItemButton>
						</List>
					</Box>
				</Box>
			</Drawer>

			{/* Fixed Toggle Button at top left */}
			<IconButton
				onClick={() => setOpen(!open)}
				sx={{
					position: 'fixed',
					top: '1rem',
					left: '1rem',
					zIndex: 1300,
					color: '#646cff',
					backgroundColor: '#1a1a1a',
					border: '1px solid #646cff',
					'&:hover': {
						backgroundColor: 'rgba(100, 108, 255, 0.1)'
					}
				}}
			>
				{open ? <CloseIcon /> : <MenuIcon />}
			</IconButton>
		</>
	);
}
