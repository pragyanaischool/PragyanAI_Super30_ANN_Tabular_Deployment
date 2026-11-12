// ==========================================
// FRONTEND: React Dashboard (App.js)
// ==========================================

import React, { useState } from 'react';
import './App.css';

function App() {
  // Pre-loaded with a standard sample clinical profile for testing
  const [formData, setFormData] = useState({
    pregnancies: '6',
    glucose: '148',
    blood_pressure: '72',
    skin_thickness: '35',
    insulin: '0',
    bmi: '33.6',
    diabetes_pedigree: '0.627',
    age: '50'
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);
    setError(null);

    // Backend URL from environment variables or fallback to deployed Render service
    const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || "https://your-backend-service.onrender.com";

    try {
      const response = await fetch(`${BACKEND_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          pregnancies: parseFloat(formData.pregnancies),
          glucose: parseFloat(formData.glucose),
          blood_pressure: parseFloat(formData.blood_pressure),
          skin_thickness: parseFloat(formData.skin_thickness),
          insulin: parseFloat(formData.insulin),
          bmi: parseFloat(formData.bmi),
          diabetes_pedigree: parseFloat(formData.diabetes_pedigree),
          age: parseFloat(formData.age)
        })
      });

      if (!response.ok) {
        throw new Error(`Server responded with status ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      console.error(err);
      setError("Failed to connect to the backend inference engine. Please check your network or backend URL.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App" style={{ fontFamily: 'Segoe UI, Tahoma, Geneva, Verdana, sans-serif', padding: '2rem', maxWidth: '650px', margin: 'auto', background: '#f8f9fa', borderRadius: '10px', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }}>
      <header style={{ textAlign: 'center', marginBottom: '2rem' }}>
        <h2 style={{ color: '#007BFF', margin: '0 0 0.5rem 0' }}>🇮🇳 India Diabetes Risk Analyzer</h2>
        <p style={{ color: '#6c757d', fontSize: '0.95rem' }}>Powered by Tiny TFLite ANN Model & FastAPI Inference Engine</p>
      </header>

      <form onSubmit={handleSubmit} style={{ display: 'grid', gap: '1rem', background: 'white', padding: '1.5rem', borderRadius: '8px', border: '1px solid #dee2e6' }}>
        {Object.keys(formData).map((key) => (
          <div key={key} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <label style={{ textTransform: 'capitalize', fontWeight: '500', color: '#495057' }}>
              {key.replace('_', ' ')}:
            </label>
            <input 
              type="number" 
              step="any" 
              name={key} 
              value={formData[key]} 
              onChange={handleChange} 
              required 
              style={{ padding: '0.5rem', width: '55%', borderRadius: '4px', border: '1px solid #ced4da', fontSize: '1rem' }}
            />
          </div>
        ))}
        <button 
          type="submit" 
          disabled={loading}
          style={{ padding: '0.75rem', background: '#007BFF', color: 'white', border: 'none', borderRadius: '4px', fontSize: '1rem', fontWeight: 'bold', cursor: 'pointer', marginTop: '1rem', transition: 'background 0.2s' }}
        >
          {loading ? 'Evaluating Patient Data...' : 'Run AI Prediction'}
        </button>
      </form>

      {error && (
        <div style={{ marginTop: '1.5rem', padding: '1rem', background: '#f8d7da', color: '#721c24', borderRadius: '6px', border: '1px solid #f5c6cb' }}>
          <strong>Error:</strong> {error}
        </div>
      )}

      {result && (
        <div style={{ marginTop: '1.5rem', padding: '1.2rem', background: '#d1e7dd', color: '#0f5132', borderRadius: '6px', border: '1px solid #badbcc' }}>
          <h3 style={{ margin: '0 0 0.8rem 0' }}>📋 Diagnostic Report</h3>
          <p style={{ margin: '0.4rem 0', fontSize: '1.05rem' }}><strong>Status:</strong> {result.diagnostic_result}</p>
          <p style={{ margin: '0.4rem 0', fontSize: '1.05rem' }}><strong>Confidence Probability:</strong> {result.confidence_percentage}</p>
        </div>
      )}
    </div>
  );
}

export default App;
