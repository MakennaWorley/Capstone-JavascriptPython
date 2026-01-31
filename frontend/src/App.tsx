import { useState } from 'react';
import DatasetModelForm from './components/DatasetModelForm.js';

export default function App() {
	const API_BASE = 'http://localhost:8000';

	const [msg, setMsg] = useState<string | null>(null);
	const [status, setStatus] = useState('');
	const [xApiKey, setXApiKey] = useState('');

	// --- Ping FastAPI ---
	async function pingBackend() {
		try {
			setStatus('Pinging...');
			const r = await fetch(`${API_BASE}/api/hello`);
			const j = await r.json();
			setMsg(j.message);
			setStatus('Success!');
		} catch {
			setStatus('Error contacting backend');
		}
	}

	return (
		<div style={{ padding: '2rem', fontFamily: 'system-ui, sans-serif' }}>
			<button onClick={pingBackend}>Ping FastAPI</button>
			{msg && <p>Message: {msg}</p>}
			{status && <p>{status}</p>}

			<h2>Auth</h2>
			<div style={{ display: 'grid', gap: '0.75rem', maxWidth: 420, marginBottom: 18 }}>
				<label>
					X-API-Key (header)
					<input
						value={xApiKey}
						onChange={(e) => setXApiKey(e.target.value)}
						placeholder="test123"
						style={{ width: '100%', padding: '0.5rem' }}
					/>
				</label>
			</div>

			<DatasetModelForm apiBase={API_BASE} xApiKey={xApiKey} />
		</div>
	);
}
