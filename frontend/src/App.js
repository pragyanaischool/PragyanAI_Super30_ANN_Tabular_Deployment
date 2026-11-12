import React, { useState } from 'react';
import './App.css';

// Automatically fallback to localhost if environment variable isn't injected yet
const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || "https://pragyanai-super30-ann-tabular-deployment.onrender.com";

function App() {
  const [formData, setFormData] = useState({
    Pregnancies: '',
    Glucose: '',
    BloodPressure: '',
    SkinThickness: '',
    Insulin: '',
    BMI: '',
    DiabetesPedigreeFunction: '',
    Age: ''
  });

  const [predictionResult, setPredictionResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setPredictionResult(null);

    // Convert string inputs to floats
    const formattedData = {
      Pregnancies: parseFloat(formData.Pregnancies),
      Glucose: parseFloat(formData.Glucose),
      BloodPressure: parseFloat(formData.BloodPressure),
      SkinThickness: parseFloat(formData.SkinThickness),
      Insulin: parseFloat(formData.Insulin),
      BMI: parseFloat(formData.BMI),
      DiabetesPedigreeFunction: parseFloat(formData.DiabetesPedigreeFunction),
      Age: parseFloat(formData.Age),
    };

    console.log(`[Frontend Log] Sending prediction request to: ${BACKEND_URL}/predict`);
    console.log("[Frontend Log] Payload data:", formattedData);

    try {
      const response = await fetch(`${BACKEND_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formattedData),
      });

      console.log(`[Frontend Log] Response status: ${response.status} ${response.statusText}`);

      if (!response.ok) {
        throw new Error(`Server responded with status: ${response.status}`);
      }

      const data = await response.json();
      console.log("[Frontend Log] Success! Received prediction data:", data);
      setPredictionResult(data);
    } catch (err) {
      console.error("[Frontend Error] Failed to connect to backend inference engine:", err);
      setError("Failed to connect to the backend inference engine. Please check your network, or note that Render free tier instances take 30-60 seconds to spin up from sleep.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App" style={{ padding: '20px', fontFamily: 'Arial' }}>
      <h2>Pima Indians Diabetes Risk Prediction</h2>
      <p>Connected to Backend: <code>{BACKEND_URL}</code></p>
      
      <form onSubmit={handleSubmit} style={{ display: 'grid', gap: '10px', maxWidth: '400px' }}>
        {Object.keys(formData).map((key) => (
          <div key={key} style={{ display: 'flex', justifyContent: 'space-between' }}>
            <label>{key}:</label>
            <input
              type="number"
              step="any"
              name={key}
              value={formData[key]}
              onChange={handleChange}
              required
            />
          </div>
        ))}
        <button type="submit" disabled={loading} style={{ padding: '10px', background: '#007BFF', color: 'white', border: 'none' }}>
          {loading ? 'Predicting (Awakening Server if Asleep)...' : 'Predict Risk'}
        </button>
      </form>

      {error && <div style={{ color: 'red', marginTop: '15px' }}>{error}</div>}

      {predictionResult && (
        <div style={{ marginTop: '20px', padding: '15px', background: '#eef', border: '1px solid #ccd' }}>
          <h3>Results:</h3>
          <p><strong>Status:</strong> {predictionResult.result}</p>
          <p><strong>Probability Score:</strong> {predictionResult.probability * 100}%</p>
        </div>
      )}
    </div>
  );
}

export default App;
