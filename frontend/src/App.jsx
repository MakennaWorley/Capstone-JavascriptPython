import { useEffect, useState } from 'react'

export default function App() {
  const [msg, setMsg] = useState('loading...')
  useEffect(() => {
    fetch(`/fastapi/api/hello`) 
      .then(r => r.json())
      .then(d => setMsg(d.message))
      .catch(() => setMsg('error'))
  }, [])
  return <h1 style={{fontFamily:'system-ui, sans-serif'}}>{msg}</h1>
}