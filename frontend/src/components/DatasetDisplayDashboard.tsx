import DownloadIcon from '@mui/icons-material/Download';
import { Button, MenuItem, Paper, Table, TableBody, TableCell, TableContainer, TableHead, TablePagination, TableRow, TextField } from '@mui/material';
import { useTheme } from '@mui/material/styles';
import { useEffect, useMemo, useRef, useState } from 'react';
import FamilyTreeVisualization from './FamilyTreeVisualization.js';
import LoadingProgress from './LoadingProgress.js';

type DatasetDashboardProps = {
	apiBase: string;
	xApiKey: string;
	selectedDataset: string;
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

export default function DatasetDashboard({ apiBase, xApiKey, selectedDataset }: DatasetDashboardProps) {
	const theme = useTheme();
	const [loading, setLoading] = useState(false);
	const [showLoadingProgress, setShowLoadingProgress] = useState(false);
	const [data, setData] = useState<DashboardState>({});
	const [selectedIndId, setSelectedIndId] = useState<string>('');
	const [familyTreeData, setFamilyTreeData] = useState<any>(null);
	const [columnPageIndex, setColumnPageIndex] = useState(0);
	const COLUMNS_PER_PAGE = 10;

	// Cache keyed by `dataset:indId` to avoid redundant backend requests
	const familyTreeCache = useRef<Map<string, any>>(new Map());

	// Show loading progress only after 1 second to avoid spasms on fast requests
	useEffect(() => {
		if (loading) {
			const timeoutId = setTimeout(() => setShowLoadingProgress(true), 1000);
			return () => clearTimeout(timeoutId);
		} else {
			setShowLoadingProgress(false);
		}
	}, [loading]);

	// Previews
	const MAX_PARSE_ROWS = 1000;

	const observedPreview = useMemo(() => {
		if (!data.observedCsvRaw) return null;
		return parseCsvPreview(data.observedCsvRaw, MAX_PARSE_ROWS);
	}, [data.observedCsvRaw]);

	const truthPreview = useMemo(() => {
		if (!data.truthCsvRaw) return null;
		return parseCsvPreview(data.truthCsvRaw, MAX_PARSE_ROWS);
	}, [data.truthCsvRaw]);

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
			familyTreeCache.current.clear();
			setFamilyTreeData(null);
			loadDashboard();
		}
	}, [selectedDataset, apiBase, xApiKey]);

	// Auto-load (or serve from cache) when individual selection changes
	useEffect(() => {
		if (!selectedDataset || !selectedIndId) return;

		const cacheKey = `${selectedDataset}:${selectedIndId}`;
		const cached = familyTreeCache.current.get(cacheKey);
		if (cached) {
			setFamilyTreeData(cached);
			return;
		}

		setLoading(true);
		fetch(`${apiBase}/api/dataset/${encodeURIComponent(selectedDataset)}/tree/${selectedIndId}`, {
			headers: { 'x-api-key': xApiKey }
		})
			.then((resp) => {
				if (!resp.ok) throw new Error(`Tree fetch failed: ${resp.status}`);
				return resp.json();
			})
			.then((j) => {
				const componentData = j.data;
				// Pre-populate cache for every individual in this connected component.
				if (componentData?.nodes) {
					for (const node of componentData.nodes) {
						const key = `${selectedDataset}:${node.id}`;
						if (!familyTreeCache.current.has(key)) {
							familyTreeCache.current.set(key, { ...componentData, focus_id: node.id });
						}
					}
				}
				setFamilyTreeData(componentData);
			})
			.catch(() => {
				// Silently fail - tree will not display
			})
			.finally(() => setLoading(false));
	}, [selectedIndId, selectedDataset, apiBase, xApiKey]);

	async function downloadAllDatasetZip() {
		if (!selectedDataset || loading) return;

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
		<div className="dashboard-wrapper">
			<LoadingProgress isLoading={showLoadingProgress} message="Fetching your data..." />

			{/* CSV preview - merged view */}
			{mergedPreview && (
			<div className="section-wrapper">
				<div className="flex-between-mb">
					<h2 className="section-heading">Genotypes</h2>
						<Button
							variant="contained"
							startIcon={<DownloadIcon />}
							onClick={downloadAllDatasetZip}
							disabled={!selectedDataset}
							size="small"
						>
							{'Download Dataset'}
						</Button>
					</div>

					<p className="context-text">
						This table shows the merged genotype data for the selected dataset. Each row is a genomic site and each column is an
						individual in the simulated population. The values represent <strong>allele dosage</strong> — the number of copies of the
						alternate allele carried at that site (0, 1, or 2). Cells highlighted in{' '}
						<span className="text-known">green</span> are <strong>known data</strong> —
						individuals whose genes were successfully sequenced and are present in the dataset. Cells highlighted in{' '}
						<span className="text-unknown">red</span> are <strong>unknown data</strong> —
						individuals intentionally left out to simulate the real-world scenario of individuals who were never sequenced. The models are
						trained only on the known data and must infer the genotypes of these missing individuals.
					</p>

					<p className="context-text">
						Use the column paginator below to scroll across individuals. Select an individual ID from the family tree section to visualize
						their pedigree and see how their relatives' genotypes inform the inference.
					</p>

					<p className="legend-label">Legend</p>

					<p className="legend">
						<span className="legend-item">
							<span className="legend-square legend-square-known" />
							<span className="text-known">Known</span>
							<span className="empty-state">= data is known</span>
						</span>
						<span className="legend-item">
							<span className="legend-square legend-square-unknown" />
							<span className="text-unknown">Unknown</span>
							<span className="empty-state">= missing or unobserved</span>
						</span>
					</p>
					{mergedPreview.headers.length > COLUMNS_PER_PAGE && (
						<TablePagination
							component="div"
							count={mergedPreview.headers.length - 1}
							page={columnPageIndex}
							onPageChange={(_, newPage) => setColumnPageIndex(newPage)}
							rowsPerPage={COLUMNS_PER_PAGE}
							onRowsPerPageChange={() => {}}
							rowsPerPageOptions={[]}
							labelDisplayedRows={({ from, to, count }) => `Columns ${from}–${to} of ${count}`}
						/>
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
			<div className="section-wrapper">
				<h2 className="section-heading">Family Tree Explorer</h2>
					<TextField
						select
						size="small"
						label="Select Individual ID"
						value={selectedIndId}
						onChange={(e) => setSelectedIndId(e.target.value)}
						sx={{ minWidth: 200 }}
					>
						<MenuItem value="">-- Choose ID --</MenuItem>
						{availableIds.map((id) => (
							<MenuItem key={id} value={id}>
								Individual {id}
							</MenuItem>
						))}
					</TextField>
					{familyTreeData && <FamilyTreeVisualization data={familyTreeData} />}
				</div>
			)}
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
	const theme = useTheme();
	const tableRef = useRef<HTMLTableElement>(null);
	const ROWS_PER_PAGE = 100;
	const [rowPage, setRowPage] = useState(0);

	// Calculate which columns to show on this page (excluding index column 0)
	const startColIdx = columnPageIndex * columnsPerPage + 1; // Start from 1 to skip index
	const endColIdx = Math.min(startColIdx + columnsPerPage, headers.length);
	const visibleHeaders = headers.slice(startColIdx, endColIdx);
	const visibleIndices = Array.from({ length: endColIdx - startColIdx }, (_, i) => startColIdx + i);

	const pagedRows = mergedRows.slice(rowPage * ROWS_PER_PAGE, (rowPage + 1) * ROWS_PER_PAGE);

	const hoverColor = theme.palette.action.hover;

	const handleMouseOver = (e: React.MouseEvent<HTMLTableElement>) => {
		const cell = (e.target as HTMLElement).closest('td, th') as HTMLElement | null;
		if (!cell || !tableRef.current) return;
		const row = cell.parentElement;
		if (!row) return;
		const colIndex = Array.from(row.children).indexOf(cell);
		tableRef.current.querySelectorAll<HTMLElement>('[data-col-h]').forEach((el) => {
			el.style.backgroundColor = '';
			delete el.dataset.colH;
		});
		if (colIndex > 0) {
			tableRef.current.querySelectorAll<HTMLElement>(`tr > *:nth-child(${colIndex + 1})`).forEach((el) => {
				el.style.backgroundColor = hoverColor;
				el.dataset.colH = '1';
			});
		}
	};

	const handleMouseLeave = () => {
		tableRef.current?.querySelectorAll<HTMLElement>('[data-col-h]').forEach((el) => {
			el.style.backgroundColor = '';
			delete el.dataset.colH;
		});
	};

	return (
		<>
			{mergedRows.length > ROWS_PER_PAGE && (
				<TablePagination
					component="div"
					count={mergedRows.length}
					page={rowPage}
					onPageChange={(_, newPage) => setRowPage(newPage)}
					rowsPerPage={ROWS_PER_PAGE}
					onRowsPerPageChange={() => {}}
					rowsPerPageOptions={[]}
					labelDisplayedRows={({ from, to, count }) => `Rows ${from}–${to} of ${count}`}
				/>
			)}
			<TableContainer component={Paper} sx={{ width: '100%', maxHeight: 400, overflow: 'auto' }}>
				<Table
					ref={tableRef}
					size="small"
					stickyHeader
					sx={{ tableLayout: 'auto' }}
					onMouseOver={handleMouseOver}
					onMouseLeave={handleMouseLeave}
				>
					<TableHead>
						<TableRow>
							<TableCell
								className="sticky-col-header"
								sx={{
									backgroundColor: theme.palette.background.paper,
								}}
								title={headers[0]}
							>
								{clampText(headers[0], 40)}
							</TableCell>
							{visibleHeaders.map((h, localIdx) => (
								<TableCell key={localIdx} sx={{ fontWeight: 'bold', whiteSpace: 'nowrap' }} title={h}>
									{clampText(h, 40)}
								</TableCell>
							))}
						</TableRow>
					</TableHead>
					<TableBody>
						{pagedRows.map((item, ridx) => (
							<TableRow key={ridx} hover>
								<TableCell
									className="sticky-col"
									sx={{
										fontWeight: 'bold',
										backgroundColor:
											theme.palette.mode === 'dark' ? theme.palette.background.paper : theme.palette.background.default,
									}}
									title={item.displayed[0] ?? ''}
								>
									{clampText(String(item.displayed[0] ?? ''), 60)}
								</TableCell>
								{visibleIndices.map((colIdx) => {
									const isKnown = item.knownMask[colIdx];
									return (
										<TableCell
											key={colIdx}
											className={isKnown ? 'cell-known' : 'cell-unknown'}
											sx={{ whiteSpace: 'nowrap' }}
											title={item.displayed[colIdx] ?? ''}
										>
											{clampText(String(item.displayed[colIdx] ?? ''), 60)}
										</TableCell>
									);
								})}
							</TableRow>
						))}
						{mergedRows.length === 0 && (
							<TableRow>
								<TableCell colSpan={(visibleHeaders.length || 0) + 1} sx={{ opacity: 0.75 }}>
									No rows to display.
								</TableCell>
							</TableRow>
						)}
					</TableBody>
				</Table>
			</TableContainer>
		</>
	);
}

function CsvTable({ title, preview, maxRows }: { title: string; preview: CsvPreview; maxRows: number }) {
	const { headers, rows, estimatedTotalRows } = preview;

	return (
		<div className="section-wrapper csv-preview-box">
			<h4 className="section-heading">{title}</h4>

			<p className="description-faint">
				Showing first <b>{Math.min(rows.length, maxRows)}</b>
				{typeof estimatedTotalRows === 'number' ? (
					<>
						{' '}
						of about <b>{estimatedTotalRows.toLocaleString()}</b> rows
					</>
				) : null}
				.
			</p>

			<div className="table-scroll">
				<table className="csv-table">
					<thead>
						<tr>
							{headers.map((h, idx) => (
								<th
									key={idx}
									className="csv-th"
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
										className="csv-td"
										title={r[cidx] ?? ''}
									>
										{clampText(String(r[cidx] ?? ''), 60)}
									</td>
								))}
							</tr>
						))}
						{rows.length === 0 && (
							<tr>
								<td colSpan={headers.length || 1} className="empty-state">
									No rows to display.
								</td>
							</tr>
						)}
					</tbody>
				</table>
			</div>

			<p className="table-note">
				Tip: these CSVs are wide (lots of <code>ind_####</code> columns). Horizontal scroll is expected.
			</p>
		</div>
	);
}
