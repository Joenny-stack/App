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

  // Function to send one sample value to the backend
  const sendSample = async () => {
    if (!newValue) return;
    const payload = { sensor_id: sensorId, value: parseFloat(newValue) };
    setLoading(true);
    try {
      const res = await fetch(`${API_URL}/sample`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();

      if (data.prediction || data.label) {
        // store raw prediction and any server-provided label
        setPrediction(data.prediction ?? data.label);
        setPredictionLabel(data.label ?? null);
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
          {prediction ? (
            (() => {
              // Try to interpret common numeric/class outputs
              // If server provided a label, prefer that
              const serverLabel = predictionLabel;
              if (serverLabel) {
                return (
                  <div>
                    <p className={`text-xl font-bold text-indigo-600`}>{serverLabel} <span className="text-sm text-gray-500">({prediction})</span></p>
                    <p className="mt-2 text-sm text-gray-600">Server-provided label</p>
                  </div>
                );
              }

              const num = Number(prediction);
              let label = prediction;
              let colorClass = "text-indigo-600";

              if (!Number.isNaN(num)) {
                if (num === 0) {
                  label = "Healthy";
                  colorClass = "text-green-600";
                } else if (num === 1) {
                  label = "Faulty";
                  colorClass = "text-red-600";
                } else {
                  // numeric but not 0/1 — show as a score
                  label = `Score: ${Number.isInteger(num) ? num : num.toFixed(3)}`;
                  colorClass = "text-indigo-600";
                }
              } else {
                // non-numeric prediction (string labels)
                label = prediction;
                colorClass = "text-indigo-600";
              }

              return (
                <div>
                  <p className={`text-xl font-bold ${colorClass}`}>{label} <span className="text-sm text-gray-500">({prediction})</span></p>
                  <p className="mt-2 text-sm text-gray-600">
                    Interpretation: {num === 0 ? '0 typically means the system is healthy.' : num === 1 ? '1 typically means a fault or unhealthy state.' : 'This value is the raw model output.'}
                  </p>
                  <p className="mt-1 text-xs text-gray-400">
                    Note: If your trained model uses different labels (for example strings or probabilities), we can surface those from the backend.
                  </p>
                </div>
              );
            })()
          ) : (
            <p className="text-gray-500 italic">No prediction yet.</p>
          )}

        {statusMsg && (
          <p className="mt-2 text-sm text-gray-600">{statusMsg}</p>
        )}

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
