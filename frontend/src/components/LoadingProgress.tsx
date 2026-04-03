
type LoadingProgressProps = {
	isLoading: boolean;
	message: string;
};

export default function LoadingProgress({ isLoading, message }: LoadingProgressProps) {
	if (!isLoading) return null;

	return (
		<div
			style={{
				marginTop: '1rem',
				padding: '1rem',
				backgroundColor: '#2a2a2a',
				borderRadius: '4px',
				border: '1px solid #646cff'
			}}
		>
			<div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
				<div style={{ display: 'flex', gap: '0.25rem' }}>
					<div
						style={{
							width: '8px',
							height: '8px',
							borderRadius: '50%',
							backgroundColor: '#646cff',
							animation: 'pulse 1.5s ease-in-out 0s infinite'
						}}
					/>
					<div
						style={{
							width: '8px',
							height: '8px',
							borderRadius: '50%',
							backgroundColor: '#646cff',
							animation: 'pulse 1.5s ease-in-out 0.3s infinite'
						}}
					/>
					<div
						style={{
							width: '8px',
							height: '8px',
							borderRadius: '50%',
							backgroundColor: '#646cff',
							animation: 'pulse 1.5s ease-in-out 0.6s infinite'
						}}
					/>
				</div>
				<span style={{ fontSize: '0.95rem', color: 'rgba(255, 255, 255, 0.87)' }}>{message}</span>
			</div>

			{/* Loading bar */}
			<div
				style={{
					marginTop: '0.75rem',
					width: '100%',
					height: '4px',
					backgroundColor: '#1a1a1a',
					borderRadius: '2px',
					overflow: 'hidden'
				}}
			>
				<div
					style={{
						height: '100%',
						backgroundColor: '#646cff',
						borderRadius: '2px',
						animation: 'shimmer 2s infinite'
					}}
				/>
			</div>

			<style>{`
				@keyframes pulse {
					0% {
						opacity: 0.4;
					}
					50% {
						opacity: 1;
					}
					100% {
						opacity: 0.4;
					}
				}

				@keyframes shimmer {
					0% {
						width: 0%;
					}
					50% {
						width: 100%;
					}
					100% {
						width: 0%;
					}
				}
			`}</style>
		</div>
	);
}
