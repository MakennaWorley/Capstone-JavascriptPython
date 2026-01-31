import { useState } from 'react';

export default function App() {
	const API_BASE = 'http://localhost:8000';

	const [msg, setMsg] = useState(null);
	const [status, setStatus] = useState('');
	const [xApiKey, setXApiKey] = useState('');
	const [datasetName, setDatasetName] = useState('');
	const [notes, setNotes] = useState('');
	const [numSamples, setNumSamples] = useState(100);

	// --- Ping FastAPI ---
	async function pingBackend() {
		try {
			setStatus('Pinging...');
			const r = await fetch(`${API_BASE}/api/hello`);
			const j = await r.json();
			setMsg(j.message);
			setStatus('Success!');
		} catch (err) {
			setStatus('Error contacting backend');
		}
	}

	// --- Create Dataset ---
	async function createDatasetStub() {
		try {
			setStatus('Sending...');

			const body = {
				datasetName,
				notes,
				numSamples
			};

			const r = await fetch(`${API_BASE}/api/create/data`, {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json',
					'X-API-Key': xApiKey
				},
				body: JSON.stringify(body)
			});

			if (!r.ok) {
				const text = await r.text().catch(() => '');
				setStatus(`Error ${r.status}: ${text}`);
				return;
			}

			setStatus('Sent OK (check FastAPI terminal for printed headers).');
		} catch (err) {
			console.error(err);
			setStatus('Network error');
		}
	}

	return (
		<div style={{ padding: '2rem', fontFamily: 'system-ui, sans-serif' }}>
			<button onClick={pingBackend}>Ping FastAPI</button>
			{msg && <p>Message: {msg}</p>}

			<h2>Create Dataset (stub)</h2>

			<div style={{ display: 'grid', gap: '0.75rem', maxWidth: 420 }}>
				<label>
					X-API-Key (header)
					<input
						value={xApiKey}
						onChange={(e) => setXApiKey(e.target.value)}
						placeholder="test123"
						style={{ width: '100%', padding: '0.5rem' }}
					/>
				</label>

				<label>
					Dataset name
					<input
						value={datasetName}
						onChange={(e) => setDatasetName(e.target.value)}
						placeholder="my-dataset"
						style={{ width: '100%', padding: '0.5rem' }}
					/>
				</label>

				<label>
					Notes
					<input
						value={notes}
						onChange={(e) => setNotes(e.target.value)}
						placeholder="optional notes..."
						style={{ width: '100%', padding: '0.5rem' }}
					/>
				</label>

				<label>
					Num samples (number)
					<input
						type="number"
						value={numSamples}
						onChange={(e) => setNumSamples(Number(e.target.value))}
						style={{ width: '100%', padding: '0.5rem' }}
					/>
				</label>

				<button onClick={createDatasetStub} style={{ padding: '0.6rem' }}>
					Send to backend
				</button>

				{status && <p>{status}</p>}
			</div>
		</div>
	);
}
