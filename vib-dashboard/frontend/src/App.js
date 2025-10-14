import React, { useEffect, useState } from "react";
import Plot from "react-plotly.js";

function App() {
  const API_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";
  const [values, setValues] = useState([]); // stores recent raw sensor values
  const [prediction, setPrediction] = useState(null);
  const [predictionLabel, setPredictionLabel] = useState(null);
  const [statusMsg, setStatusMsg] = useState(""); // show buffering / error messages
  const [bufferLen, setBufferLen] = useState(0);
  const BUFFER_SIZE = 10; // reduced for local testing; change back to 256 for production
  const [bufferSize, setBufferSize] = useState(BUFFER_SIZE);
  const [sensorId, setSensorId] = useState("esp1");
  const [newValue, setNewValue] = useState("");
  const [loading, setLoading] = useState(false);
  const [labelMap, setLabelMap] = useState({0: 'Healthy', 1: 'Degraded', 2: 'Faulty'});
  // RUL client state
  const [rul, setRul] = useState(null);
  const [rulLoading, setRulLoading] = useState(false);
  const [rulHistory, setRulHistory] = useState([]);

  // Function to send one sample value to the backend
  const sendSample = async () => {
    if (!newValue) return;
    // include client timestamp (UTC seconds) so server can compute accurate durations
    const payload = { sensor_id: sensorId, value: parseFloat(newValue), timestamp: Math.floor(Date.now() / 1000) };
    setLoading(true);
    try {
      const res = await fetch(`${API_URL}/sample`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();

      // Treat numeric 0 as a valid prediction (don't use truthiness)
      if (data.prediction !== undefined && data.prediction !== null) {
        // store raw prediction; ignore any server-provided label (handled client-side)
        setPrediction(data.prediction);
        setPredictionLabel(null);
        setStatusMsg("✅ Prediction ready");
        setBufferLen(0);
      } else if (data.status === "buffering") {
        // prefer server-provided buffer_size if available
        const srvBuf = typeof data.buffer_size === 'number' ? data.buffer_size : (data.buffer_size ? parseInt(data.buffer_size) : BUFFER_SIZE);
        setBufferSize(srvBuf);
        setStatusMsg(`Buffering: ${data.buffer_len} / ${srvBuf} samples collected`);
        setBufferLen(typeof data.buffer_len === 'number' ? data.buffer_len : parseInt(data.buffer_len || '0'));
      } else if (data.error) {
        setStatusMsg(`⚠️ Error: ${data.error}`);
          setBufferLen(0);
      }

      setValues((prev) => [...prev.slice(-99), parseFloat(newValue)]);
      setNewValue("");
      // Request updated RUL from the backend (fire-and-forget)
      predictRUL().catch((e) => console.error('predictRUL failed', e));
    } catch (err) {
      console.error("Send error:", err);
      setStatusMsg("⚠️ Connection or server error");
    } finally {
      setLoading(false);
    }
  };

  // Plot trace for raw sensor values
  const trace = {
    x: values.map((_, i) => i),
    y: values,
    type: "scatter",
    mode: "lines+markers",
    line: { shape: "spline" },
    marker: { size: 4 },
    name: "Sensor Values",
  };

  // fetch label map from server on mount
  useEffect(() => {
    (async () => {
      try {
        const res = await fetch(`${API_URL}/label-map`);
        const data = await res.json();
        if (data && data.label_map) {
          setLabelMap(data.label_map);
        }
      } catch (e) {
        // ignore and use defaults
      }
    })();
  }, []);

  // Fetch RUL from backend and update state
  const predictRUL = async () => {
    setRulLoading(true);
    try {
      const q = new URLSearchParams({ sensor_id: sensorId }).toString();
      const res = await fetch(`${API_URL}/rul?${q}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      if (data) {
        setRul(data);
        // prefer mean if present, else try 'rul'
        const meanVal = (typeof data.rul_mean === 'number') ? data.rul_mean : (typeof data.rul === 'number' ? data.rul : null);
        if (meanVal !== null) {
          setRulHistory((prev) => [...prev.slice(-99), Number(meanVal)]);
        }
        setStatusMsg("✅ RUL updated");
      }
    } catch (e) {
      console.error('RUL prediction error:', e);
      setStatusMsg('⚠️ RUL fetch failed');
    } finally {
      setRulLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 text-gray-800 flex flex-col items-center p-6">
      <div className="max-w-3xl w-full bg-white shadow-xl rounded-2xl p-6">
        <h1 className="text-2xl font-bold mb-4 text-center text-indigo-600">
          Health Monitoring Dashboard
        </h1>

        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-6">
          <div className="flex gap-2 items-center">
            <label className="font-semibold">Sensor ID:</label>
            <input
              type="text"
              value={sensorId}
              onChange={(e) => setSensorId(e.target.value)}
              className="border px-2 py-1 rounded-md"
            />
          </div>
          <div className="flex gap-2 items-center">
            <input
              type="number"
              value={newValue}
              onChange={(e) => setNewValue(e.target.value)}
              placeholder="Enter sensor value"
              className="border px-2 py-1 rounded-md"
            />
            <button
              onClick={sendSample}
              disabled={loading}
              className="bg-indigo-600 hover:bg-indigo-700 text-white font-semibold px-4 py-2 rounded-md disabled:bg-gray-400"
            >
              {loading ? "Sending..." : "Send"}
            </button>
          </div>
        </div>

        {/* Status / Prediction Section */}
        <div className="bg-gray-50 rounded-md p-4 mb-6">
          <h2 className="text-lg font-semibold text-gray-700 mb-2">
            Health State Prediction:
          </h2>
          {prediction !== null && prediction !== undefined ? (
            (() => {
              // Try to interpret common numeric/class outputs
              const num = Number(prediction);
              let label = prediction;
              let colorClass = "text-indigo-600";
              let activeIndex = null;

              // Map numeric outputs using labelMap from server (or fallback)
              if (!Number.isNaN(num)) {
                const mapped = Object.prototype.hasOwnProperty.call(labelMap, num) ? labelMap[num] : null;
                if (mapped) {
                  label = mapped;
                  // choose color by mapping key
                  if (num === 0) colorClass = "text-green-600";
                  else if (num === 1) colorClass = "text-yellow-600";
                  else if (num === 2) colorClass = "text-red-600";
                  else colorClass = "text-indigo-600";
                  activeIndex = [0,1,2].includes(num) ? num : null;
                } else {
                  label = `Score: ${Number.isInteger(num) ? num : num.toFixed(3)}`;
                  colorClass = "text-indigo-600";
                }
              } else {
                // non-numeric prediction (string labels)
                label = prediction;
                colorClass = "text-indigo-600";
              }

              // state scale visualization
              const states = [
                { key: 0, label: 'Healthy', color: 'bg-green-500' },
                { key: 1, label: 'Degraded', color: 'bg-yellow-500' },
                { key: 2, label: 'Faulty', color: 'bg-red-500' },
              ];

              return (
                <div>
                  <p className={`text-xl font-bold ${colorClass}`}>{label} <span className="text-sm text-gray-500">({prediction})</span></p>

                  <div className="mt-3">
                    <div className="w-full bg-gray-200 rounded-full h-8 overflow-hidden flex">
                      {states.map((s, i) => (
                        <div
                          key={s.key}
                          className={`${s.color} flex-1 flex items-center justify-center text-white text-sm font-semibold ${activeIndex === i ? 'ring-2 ring-offset-1 ring-indigo-300' : 'opacity-80'}`}
                        >
                          {s.label}
                        </div>
                      ))}
                    </div>
                    <p className="mt-2 text-sm text-gray-600">
                      Interpretation: {activeIndex === 0 ? '0 = Healthy' : activeIndex === 1 ? '1 = Degraded' : activeIndex === 2 ? '2 = Faulty' : 'This value is the raw model output.'}
                    </p>
                  </div>
                </div>
              );
            })()
          ) : (
            <p className="text-gray-500 italic">No prediction yet.</p>
          )}

        {statusMsg && (
          <p className="mt-2 text-sm text-gray-600">{statusMsg}</p>
        )}

        {/* RUL Display Card */}
        <div className="mt-4">
          <div className="bg-white border rounded-md p-4">
            <h3 className="text-md font-semibold text-gray-700">Remaining Useful Life</h3>
            {rul ? (
              <div className="mt-2 grid grid-cols-1 sm:grid-cols-3 gap-3 items-center">
                <div>
                  <p className="text-sm text-gray-500">Mean</p>
                  <p className="text-xl font-bold text-indigo-600">{typeof rul.rul_mean === 'number' ? rul.rul_mean.toFixed(2) : String(rul.rul_mean ?? '—')} <span className="text-sm text-gray-500">{rul.units ?? ''}</span></p>
                </div>
                <div>
                  <p className="text-sm text-gray-500">Upper</p>
                  <p className="text-xl font-bold text-green-600">{typeof rul.rul_upper === 'number' ? rul.rul_upper.toFixed(2) : String(rul.rul_upper ?? '—')} <span className="text-sm text-gray-500">{rul.units ?? ''}</span></p>
                </div>
                <div>
                  <p className="text-sm text-gray-500">Lower</p>
                  <p className="text-xl font-bold text-red-600">{typeof rul.rul_lower === 'number' ? rul.rul_lower.toFixed(2) : String(rul.rul_lower ?? '—')} <span className="text-sm text-gray-500">{rul.units ?? ''}</span></p>
                </div>

                {rulHistory.length > 0 && (
                  <div className="sm:col-span-3 mt-2">
                    <Plot
                      data={[{ x: rulHistory.map((_, i) => i), y: rulHistory, type: 'scatter', mode: 'lines+markers', line: { color: '#6366F1' } }]}
                      layout={{ width: '100%', height: 140, margin: { t: 10, b: 30, l: 30, r: 10 }, paper_bgcolor: 'transparent', plot_bgcolor: 'transparent', showlegend:false }}
                      style={{ width: '100%' }}
                      useResizeHandler
                    />
                  </div>
                )}
              </div>
            ) : (
              <p className="text-sm text-gray-500 mt-2">No RUL prediction yet.</p>
            )}
          </div>
        </div>

        {/* Progress bar */}
        {bufferLen > 0 && (
          (() => {
            const effectiveSize = bufferSize || BUFFER_SIZE;
            const progress = Math.max(0, Math.min(100, (bufferLen / effectiveSize) * 100));
            return (
              <div className="mt-3 w-full bg-gray-200 rounded-full h-2.5">
                <div
                  className="bg-indigo-600 h-2.5 rounded-full transition-all duration-300"
                  style={{ width: `${progress}%` }}
                ></div>
              </div>
            );
          })()
        )}

        </div>

        {/* Chart Section */}
        <div>
          <h2 className="text-lg font-semibold text-gray-700 mb-2">
            Raw Vibration Signals
          </h2>
          <Plot
            data={[trace]}
            layout={{
              width: "100%",
              height: 400,
              title: "Live Sensor Values",
              margin: { t: 40, r: 20, l: 50, b: 40 },
              paper_bgcolor: "#fff",
              plot_bgcolor: "#f9fafb",
            }}
            style={{ width: "100%" }}
            useResizeHandler
          />
        </div>
      </div>
    </div>
  );
}

export default App;
