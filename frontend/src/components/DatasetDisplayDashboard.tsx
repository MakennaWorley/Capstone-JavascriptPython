import { useEffect, useMemo, useState } from 'react';
import FamilyTreeVisualization from './FamilyTreeVisualization.js';
import LoadingProgress from './LoadingProgress.js';

type DatasetDashboardProps = {
	apiBase: string;
	xApiKey: string;
	selectedDataset: string;
	maxPreviewRows?: number;
};

type CsvPreview = {
	headers: string[];
	rows: string[][];
	estimatedTotalRows?: number;
};

type DashboardState = {
	observedCsvRaw?: string;
	truthCsvRaw?: string;
};

type ApiEnvelope =
	| { status: 'success'; data?: unknown; files?: unknown; message?: string }
	| { status: 'error'; message?: string; code?: string }
	| unknown;

function parseCsvPreview(csvText: string, maxRows: number): CsvPreview {
	const text = csvText.replace(/^\uFEFF/, ''); // strip BOM if present
	const lines: string[] = [];
	// Normalize newlines but keep it simple
	text.split(/\r\n|\n|\r/).forEach((l) => {
		// keep empty lines out
		if (l.trim().length > 0) lines.push(l);
	});

	if (lines.length === 0) return { headers: [], rows: [] };

	// Robust-ish CSV line parser (handles quoted commas)
	const parseLine = (line: string): string[] => {
		const out: string[] = [];
		let cur = '';
		let inQuotes = false;

		for (let i = 0; i < line.length; i++) {
			const ch = line[i];

			if (ch === '"') {
				// If we see "" inside quotes, that's an escaped quote
				if (inQuotes && line[i + 1] === '"') {
					cur += '"';
					i++;
				} else {
					inQuotes = !inQuotes;
				}
				continue;
			}

			if (ch === ',' && !inQuotes) {
				out.push(cur);
				cur = '';
				continue;
			}

			cur += ch;
		}
		out.push(cur);
		return out.map((s) => s.trim());
	};

	const headers = parseLine(lines[0]);
	const rows: string[][] = [];

	for (let i = 1; i < lines.length && rows.length < maxRows; i++) {
		rows.push(parseLine(lines[i]));
	}

	return {
		headers,
		rows,
		estimatedTotalRows: Math.max(0, lines.length - 1)
	};
}

function clampText(s: string, maxLen = 80): string {
	if (s.length <= maxLen) return s;
	return `${s.slice(0, maxLen - 1)}…`;
}

export default function DatasetDashboard({ apiBase, xApiKey, selectedDataset, maxPreviewRows = 10 }: DatasetDashboardProps) {
	const [loading, setLoading] = useState(false);
	const [showLoadingProgress, setShowLoadingProgress] = useState(false);
	const [data, setData] = useState<DashboardState>({});
	const [selectedIndId, setSelectedIndId] = useState<string>('');
	const [familyTreeData, setFamilyTreeData] = useState<any>(null);
	const [columnPageIndex, setColumnPageIndex] = useState(0);
	const COLUMNS_PER_PAGE = 10;

	// Show loading progress only after 1 second to avoid spasms on fast requests
	useEffect(() => {
		if (loading) {
			const timeoutId = setTimeout(() => setShowLoadingProgress(true), 1000);
			return () => clearTimeout(timeoutId);
		} else {
			setShowLoadingProgress(false);
		}
	}, [loading]);

	const canLoad = selectedDataset.trim().length > 0 && !loading;
	const hasLoadedDashboard = !!(data.observedCsvRaw || data.truthCsvRaw);

	// Previews
	const observedPreview = useMemo(() => {
		if (!data.observedCsvRaw) return null;
		return parseCsvPreview(data.observedCsvRaw, maxPreviewRows);
	}, [data.observedCsvRaw, maxPreviewRows]);

	const truthPreview = useMemo(() => {
		if (!data.truthCsvRaw) return null;
		return parseCsvPreview(data.truthCsvRaw, maxPreviewRows);
	}, [data.truthCsvRaw, maxPreviewRows]);

	// Merged preview for overlay table
	const mergedPreview = useMemo(() => {
		if (!observedPreview || !truthPreview) return null;

		// Map truth rows by individual ID (first column) for quick lookup
		const truthMap = new Map<string, string[]>();
		truthPreview.rows.forEach((row) => {
			if (row[0]) {
				truthMap.set(row[0], row);
			}
		});

		// Merge observed and truth data
		// displayValue = observed if present, otherwise truth
		// knownMask = true only if observed is present AND matches truth
		const mergedRows = observedPreview.rows.map((obsRow) => {
			const truthRow = truthMap.get(obsRow[0]) || [];
			const displayRow: string[] = [];
			const knownMask: boolean[] = [];

			observedPreview.headers.forEach((_, idx) => {
				const obsVal = obsRow[idx] || '';
				const truthVal = truthRow[idx] || '';

				// Display: show observed if present, otherwise show truth
				displayRow.push(obsVal.trim() !== '' ? obsVal : truthVal);

				// Known only if observed is present AND matches truth
				knownMask.push(obsVal.trim() !== '' && obsVal === truthVal);
			});

			return { displayed: displayRow, knownMask };
		});

		return { headers: observedPreview.headers, mergedRows, estimatedTotalRows: observedPreview.estimatedTotalRows };
	}, [observedPreview, truthPreview]);

	// Individual selection
	const availableIds = useMemo(() => {
		if (!data.observedCsvRaw) return [];
		const firstLine = data.observedCsvRaw.split('\n')[0] || '';
		// Match headers like i_0000, i_0001 and convert to "0", "1"
		return firstLine
			.split(',')
			.map((h) => h.trim())
			.filter((h) => h.startsWith('i_'))
			.map((h) => h.replace('i_', '').replace(/^0+/, '') || '0');
	}, [data.observedCsvRaw]);

	async function loadDashboard() {
		if (!selectedDataset) return;

		setLoading(true);
		setData({});

		try {
			const url = `${apiBase}/api/dataset/${encodeURIComponent(selectedDataset)}/dashboard`;
			const resp = await fetch(url, {
				method: 'GET',
				headers: {
					'x-api-key': xApiKey
				}
			});

			const contentType = resp.headers.get('content-type') || '';

			if (!resp.ok) {
				return;
			}

			// JSON envelope
			if (contentType.includes('application/json')) {
				const j = (await resp.json()) as ApiEnvelope;

				// Accept multiple possible shapes:
				// A) { status:'success', data:{ observed_genotypes_csv:'...', truth_genotypes_csv:'...', trees_base64:'...' } }
				// B) { status:'success', files:{ ... } }
				// C) { observed_genotypes_csv:'...', truth_genotypes_csv:'...', trees_base64:'...' } (no envelope)
				const payload = j?.data ?? j?.files ?? j;

				const next: DashboardState = {};

				// Try common keys
				next.observedCsvRaw = payload?.observed_genotypes_csv ?? payload?.observed_csv ?? payload?.observedGenotypesCsv ?? payload?.observed;

				next.truthCsvRaw = payload?.truth_genotypes_csv ?? payload?.truth_csv ?? payload?.truthGenotypesCsv ?? payload?.truth;

				setData(next);
				return;
			}

			const txt = await resp.text();
			setData({
				observedCsvRaw: txt
			});
		} catch (e: any) {
			// Silently fail - LoadingProgress will hide automatically
		} finally {
			setLoading(false);
		}
	}

	// Automatically load dashboard when dataset selection changes
	useEffect(() => {
		if (selectedDataset) {
			loadDashboard();
		}
	}, [selectedDataset, apiBase, xApiKey]);

	async function loadFamilyTree() {
		if (!selectedDataset || !selectedIndId) return;

		setLoading(true);
		try {
			const url = `${apiBase}/api/dataset/${encodeURIComponent(selectedDataset)}/tree/${selectedIndId}`;
			const resp = await fetch(url, {
				headers: { 'x-api-key': xApiKey }
			});

			if (!resp.ok) throw new Error(`Tree fetch failed: ${resp.status}`);

			const j = await resp.json();
			setFamilyTreeData(j.data);
		} catch (e: any) {
			// Silently fail - tree will not display
		} finally {
			setLoading(false);
		}
	}

	async function downloadAllDatasetZip() {
		if (!selectedDataset) return;

		setLoading(true);

		try {
			const url = `${apiBase}/api/dataset/${encodeURIComponent(selectedDataset)}/download`;

			const resp = await fetch(url, {
				method: 'GET',
				headers: {
					'x-api-key': xApiKey
				}
			});

			if (!resp.ok) {
				return;
			}

			const blob = await resp.blob();

			// Try to respect Content-Disposition filename=...
			const dispo = resp.headers.get('content-disposition') || '';
			const match = dispo.match(/filename\*?=(?:UTF-8''|")?([^\";\n]+)\"?/i);
			const filename = (match?.[1] ? decodeURIComponent(match[1]) : `${selectedDataset}.zip`).replace(/[/\\]/g, '_');

			const href = URL.createObjectURL(blob);
			const a = document.createElement('a');
			a.href = href;
			a.download = filename;
			document.body.appendChild(a);
			a.click();
			a.remove();
			URL.revokeObjectURL(href);
		} catch (e: any) {
			// Silently fail - LoadingProgress will hide automatically
		} finally {
			setLoading(false);
		}
	}

	return (
		<div style={{ marginTop: '1.25rem', width: '100%', maxWidth: '1400px', margin: '1.25rem auto 0 auto' }}>
			<LoadingProgress isLoading={showLoadingProgress} message="Fetching your data..." />

			{/* CSV preview - merged view */}
			{mergedPreview && (
				<div
					style={{
						marginTop: '1rem',
						padding: '0.9rem',
						border: '1px solid #ddd',
						borderRadius: 10,
						width: '100%',
						maxWidth: '100%',
						boxSizing: 'border-box',
						overflow: 'hidden'
					}}
				>
					<div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
						<h4 style={{ marginTop: 0 }}>Genotypes (preview)</h4>
						<button
							type="button"
							onClick={downloadAllDatasetZip}
							disabled={loading || !selectedDataset}
							title="Download all dataset files"
							style={{
								display: 'flex',
								alignItems: 'center',
								gap: '0.4rem',
								padding: '0.4rem 0.8rem',
								backgroundColor: '#646cff',
								color: 'white',
								border: 'none',
								borderRadius: '4px',
								cursor: 'pointer',
								fontSize: '0.9rem',
								transition: 'background-color 0.2s'
							}}
							onMouseOver={(e) => (e.currentTarget.style.backgroundColor = '#747eff')}
							onMouseOut={(e) => (e.currentTarget.style.backgroundColor = '#646cff')}
						>
							<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18" fill="currentColor">
								<path d="M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z" />
							</svg>
							{loading ? 'Preparing…' : 'Download Dataset'}
						</button>
					</div>

					<p style={{ marginTop: 0, opacity: 0.8 }}>
						<b>Green Text</b> = known data (matches truth) | <b>Red Text</b> = inferred/missing data
						<br />
						Showing first <b>{Math.min(mergedPreview.mergedRows.length, maxPreviewRows)}</b>
						{typeof mergedPreview.estimatedTotalRows === 'number' ? (
							<>
								{' '}
								of about <b>{mergedPreview.estimatedTotalRows.toLocaleString()}</b> rows
							</>
						) : null}
						.
					</p>

					{/* Column pagination controls */}
					{mergedPreview.headers.length > COLUMNS_PER_PAGE && (
						<div style={{ marginBottom: '1rem', display: 'flex', gap: '0.5rem', alignItems: 'center', width: '100%', boxSizing: 'border-box' }}>
							<button
								type="button"
								onClick={() => setColumnPageIndex(Math.max(0, columnPageIndex - 1))}
								disabled={columnPageIndex === 0}
								style={{ padding: '0.4rem 0.8rem', width: '90px', flexShrink: 0, cursor: 'pointer' }}
							>
								← Previous
							</button>
							<span style={{ minWidth: '200px', textAlign: 'center', fontSize: '0.9rem', flexShrink: 0 }}>
								Columns {columnPageIndex * COLUMNS_PER_PAGE + 2}–
								{Math.min((columnPageIndex + 1) * COLUMNS_PER_PAGE + 1, mergedPreview.headers.length)} of{' '}
								{mergedPreview.headers.length - 1}
							</span>
							<button
								type="button"
								onClick={() => setColumnPageIndex(columnPageIndex + 1)}
								disabled={(columnPageIndex + 1) * COLUMNS_PER_PAGE + 1 >= mergedPreview.headers.length}
								style={{ padding: '0.4rem 0.8rem', width: '70px', flexShrink: 0, cursor: 'pointer' }}
							>
								Next →
							</button>
						</div>
					)}

					<GenotypeTable
						headers={mergedPreview.headers}
						mergedRows={mergedPreview.mergedRows}
						columnPageIndex={columnPageIndex}
						columnsPerPage={COLUMNS_PER_PAGE}
					/>
				</div>
			)}

			{/* Family tree */}
			{data.observedCsvRaw && (
				<div
					style={{
						marginTop: '1.5rem',
						padding: '1rem',
						border: '2px solid #3b82f6',
						borderRadius: 10,
						width: '100%',
						boxSizing: 'border-box',
						overflow: 'hidden'
					}}
				>
					<h4>Family Tree Explorer</h4>
					<div style={{ display: 'flex', gap: '1rem', alignItems: 'flex-end' }}>
						<label>
							<span style={{ display: 'block', fontSize: '0.8rem' }}>Select Individual ID</span>
							<select
								value={selectedIndId}
								onChange={(e) => setSelectedIndId(e.target.value)}
								style={{ padding: '0.4rem', minWidth: '150px' }}
							>
								<option value="">-- Choose ID --</option>
								{availableIds.map((id) => (
									<option key={id} value={id}>
										Individual {id}
									</option>
								))}
							</select>
						</label>
						<button type="button" onClick={loadFamilyTree} disabled={!selectedIndId || loading}>
							Visualize Tree
						</button>
					</div>
				</div>
			)}

			{familyTreeData && <FamilyTreeVisualization data={familyTreeData} />}
		</div>
	);
}

function GenotypeTable({
	headers,
	mergedRows,
	columnPageIndex,
	columnsPerPage
}: {
	headers: string[];
	mergedRows: Array<{ displayed: string[]; knownMask: boolean[] }>;
	columnPageIndex: number;
	columnsPerPage: number;
}) {
	// Calculate which columns to show on this page (excluding index column 0)
	const startColIdx = columnPageIndex * columnsPerPage + 1; // Start from 1 to skip index
	const endColIdx = Math.min(startColIdx + columnsPerPage, headers.length);
	const visibleHeaders = headers.slice(startColIdx, endColIdx);
	const visibleIndices = Array.from({ length: endColIdx - startColIdx }, (_, i) => startColIdx + i);

	return (
		<div style={{ overflowX: 'hidden', width: '100%', boxSizing: 'border-box', minHeight: '200px' }}>
			<table style={{ borderCollapse: 'collapse', width: '100%', tableLayout: 'auto' }}>
				<thead>
					<tr>
						{/* Sticky Index Header */}
						<th
							style={{
								textAlign: 'left',
								borderBottom: '1px solid #ccc',
								padding: '0.5rem',
								whiteSpace: 'nowrap',
								position: 'sticky',
								left: 0,
								backgroundColor: '#1a1a1a',
								zIndex: 10,
								fontWeight: 'bold',
								minWidth: '100px'
							}}
							title={headers[0]}
						>
							{clampText(headers[0], 40)}
						</th>
						{/* Data Headers */}
						{visibleHeaders.map((h, localIdx) => (
							<th
								key={localIdx}
								style={{
									textAlign: 'left',
									borderBottom: '1px solid #ccc',
									padding: '0.5rem',
									whiteSpace: 'nowrap',
									fontWeight: 'bold'
								}}
								title={h}
							>
								{clampText(h, 40)}
							</th>
						))}
					</tr>
				</thead>
				<tbody>
					{mergedRows.map((item, ridx) => (
						<tr key={ridx}>
							{/* Sticky Index Column */}
							<td
								style={{
									borderBottom: '1px solid #eee',
									padding: '0.5rem',
									whiteSpace: 'nowrap',
									fontWeight: 'bold',
									position: 'sticky',
									left: 0,
									backgroundColor: '#0d0d0d',
									zIndex: 9
								}}
								title={item.displayed[0] ?? ''}
							>
								{clampText(String(item.displayed[0] ?? ''), 60)}
							</td>
							{/* Data Columns */}
							{visibleIndices.map((colIdx) => {
								const isKnown = item.knownMask[colIdx];
								return (
									<td
										key={colIdx}
										style={{
											borderBottom: '1px solid #eee',
											padding: '0.5rem',
											whiteSpace: 'nowrap',
											fontWeight: isKnown ? 'bold' : 'normal',
											color: isKnown ? '#00aa00' : '#ff6b6b' // Green for known, red for inferred
										}}
										title={item.displayed[colIdx] ?? ''}
									>
										{clampText(String(item.displayed[colIdx] ?? ''), 60)}
									</td>
								);
							})}
						</tr>
					))}
					{mergedRows.length === 0 && (
						<tr>
							<td colSpan={(visibleHeaders.length || 0) + 1} style={{ padding: '0.5rem', opacity: 0.75 }}>
								No rows to display.
							</td>
						</tr>
					)}
				</tbody>
			</table>
		</div>
	);
}

function CsvTable({ title, preview, maxRows }: { title: string; preview: CsvPreview; maxRows: number }) {
	const { headers, rows, estimatedTotalRows } = preview;

	return (
		<div style={{ marginTop: '1rem', padding: '0.9rem', border: '1px solid #ddd', borderRadius: 10 }}>
			<h4 style={{ marginTop: 0 }}>{title}</h4>

			<p style={{ marginTop: 0, opacity: 0.8 }}>
				Showing first <b>{Math.min(rows.length, maxRows)}</b>
				{typeof estimatedTotalRows === 'number' ? (
					<>
						{' '}
						of about <b>{estimatedTotalRows.toLocaleString()}</b> rows
					</>
				) : null}
				.
			</p>

			<div style={{ overflowX: 'auto' }}>
				<table style={{ borderCollapse: 'collapse', width: '100%' }}>
					<thead>
						<tr>
							{headers.map((h, idx) => (
								<th
									key={idx}
									style={{
										textAlign: 'left',
										borderBottom: '1px solid #ccc',
										padding: '0.5rem',
										whiteSpace: 'nowrap'
									}}
									title={h}
								>
									{clampText(h, 40)}
								</th>
							))}
						</tr>
					</thead>
					<tbody>
						{rows.map((r, ridx) => (
							<tr key={ridx}>
								{headers.map((_, cidx) => (
									<td
										key={cidx}
										style={{
											borderBottom: '1px solid #eee',
											padding: '0.5rem',
											whiteSpace: 'nowrap'
										}}
										title={r[cidx] ?? ''}
									>
										{clampText(String(r[cidx] ?? ''), 60)}
									</td>
								))}
							</tr>
						))}
						{rows.length === 0 && (
							<tr>
								<td colSpan={headers.length || 1} style={{ padding: '0.5rem', opacity: 0.75 }}>
									No rows to display.
								</td>
							</tr>
						)}
					</tbody>
				</table>
			</div>

			<p style={{ marginBottom: 0, marginTop: '0.75rem', opacity: 0.75 }}>
				Tip: these CSVs are wide (lots of <code>ind_####</code> columns). Horizontal scroll is expected.
			</p>
		</div>
	);
}
