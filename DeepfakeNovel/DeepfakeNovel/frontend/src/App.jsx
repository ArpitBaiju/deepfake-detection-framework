import React, { useState, useEffect } from "react";
import Header from "./components/Header";
import ImageUpload from "./components/ImageUpload";
import PredictionCard from "./components/PredictionCard";
import ComparisonTable from "./components/ComparisonTable";
import { Sparkles, Layers, Sliders, Play } from "lucide-react";

export default function App() {
  const [systemStatus, setSystemStatus] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [predictionData, setPredictionData] = useState(null);
  const [selectedModelTab, setSelectedModelTab] = useState("both"); // "both", "v1", "v2"
  const [errorMsg, setErrorMsg] = useState(null);

  // Check backend health on mount
  useEffect(() => {
    fetch("http://127.0.0.1:8000/api/health")
      .then((res) => res.json())
      .then((data) => setSystemStatus(data))
      .catch((err) => {
        console.error("Failed to connect to backend:", err);
        setSystemStatus({ status: "disconnected", device: "N/A" });
      });
  }, []);

  const handleImageSelected = (file) => {
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setPredictionData(null);
    setErrorMsg(null);
  };

  const handleRunInference = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setErrorMsg(null);

    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("model_version", selectedModelTab);

    try {
      const res = await fetch("http://127.0.0.1:8000/api/predict", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        throw new Error(`Server returned ${res.status}: ${res.statusText}`);
      }

      const data = await res.json();
      setPredictionData(data.results);
    } catch (err) {
      console.error("Prediction error:", err);
      setErrorMsg("Failed to run inference. Please ensure the Python backend server is running on port 8000.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <Header systemStatus={systemStatus} />

      <main style={{ maxWidth: "1280px", margin: "2rem auto", padding: "0 1.5rem" }}>
        {/* Top Control Bar */}
        <div style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: "1.5rem",
          background: "var(--bg-card)",
          padding: "1rem 1.5rem",
          borderRadius: "12px",
          border: "1px solid var(--border-color)"
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
            <Sliders size={18} color="#60a5fa" />
            <span style={{ fontSize: "0.9rem", fontWeight: "600" }}>Inference Target:</span>
            <div style={{ display: "flex", gap: "0.5rem" }}>
              {[
                { id: "both", label: "Dual Comparative (V1 vs V2)" },
                { id: "v1", label: "DAF-Net V1 Only" },
                { id: "v2", label: "DAF-Net V2 Only" }
              ].map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setSelectedModelTab(tab.id)}
                  style={{
                    background: selectedModelTab === tab.id ? "var(--accent-blue)" : "rgba(255, 255, 255, 0.05)",
                    color: selectedModelTab === tab.id ? "#fff" : "var(--text-muted)",
                    border: "none",
                    padding: "0.4rem 0.8rem",
                    borderRadius: "6px",
                    cursor: "pointer",
                    fontSize: "0.8rem",
                    fontWeight: "600",
                    transition: "all 0.2s ease"
                  }}
                >
                  {tab.label}
                </button>
              ))}
            </div>
          </div>

          <button
            onClick={handleRunInference}
            disabled={!selectedFile || loading}
            style={{
              display: "flex",
              alignItems: "center",
              gap: "0.5rem",
              background: selectedFile && !loading ? "linear-gradient(135deg, #3b82f6, #8b5cf6)" : "rgba(255, 255, 255, 0.1)",
              color: selectedFile && !loading ? "#fff" : "rgba(255, 255, 255, 0.4)",
              border: "none",
              padding: "0.6rem 1.5rem",
              borderRadius: "8px",
              cursor: selectedFile && !loading ? "pointer" : "not-allowed",
              fontWeight: "700",
              fontSize: "0.9rem",
              boxShadow: selectedFile && !loading ? "0 0 15px rgba(59, 130, 246, 0.4)" : "none",
              transition: "all 0.2s ease"
            }}
          >
            <Play size={16} fill="currentColor" />
            {loading ? "Analyzing Artifacts..." : "Run Inspection"}
          </button>
        </div>

        {errorMsg && (
          <div style={{
            background: "rgba(239, 68, 68, 0.15)",
            border: "1px solid var(--accent-red)",
            color: "#fca5a5",
            padding: "0.75rem 1rem",
            borderRadius: "8px",
            marginBottom: "1.5rem",
            fontSize: "0.85rem"
          }}>
            {errorMsg}
          </div>
        )}

        {/* Workspace Grid */}
        <div style={{
          display: "grid",
          gridTemplateColumns: predictionData ? "360px 1fr" : "1fr",
          gap: "1.5rem",
          alignItems: "start"
        }}>
          {/* Upload panel */}
          <ImageUpload
            onImageSelected={handleImageSelected}
            previewUrl={previewUrl}
            loading={loading}
          />

          {/* Predictions Panel */}
          {predictionData && (
            <div style={{
              display: "grid",
              gridTemplateColumns: selectedModelTab === "both" ? "1fr 1fr" : "1fr",
              gap: "1.5rem"
            }}>
              {predictionData.v1 && (
                <PredictionCard
                  versionKey="v1"
                  data={predictionData.v1}
                  activeTab={selectedModelTab}
                />
              )}
              {predictionData.v2 && (
                <PredictionCard
                  versionKey="v2"
                  data={predictionData.v2}
                  activeTab={selectedModelTab}
                />
              )}
            </div>
          )}
        </div>

        {/* Academic Comparison Table for Tomorrow's Review */}
        <ComparisonTable />
      </main>
    </div>
  );
}
