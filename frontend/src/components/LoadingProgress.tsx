type LoadingProgressProps = {
	isLoading: boolean;
	message: string;
};

export default function LoadingProgress({ isLoading, message }: LoadingProgressProps) {
	if (!isLoading) return null;

	return (
		<div className="loading-container" role="status" aria-live="polite">
			<div className="loading-dots">
				<div className="loading-dots-inner">
					<div className="loading-dot" />
					<div className="loading-dot" />
					<div className="loading-dot" />
				</div>
				<span className="loading-message">{message}</span>
			</div>
			<div className="loading-bar-track">
				<div className="loading-bar-fill" />
			</div>
		</div>
	);
}
