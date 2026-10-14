import React from "react";
import { CheckCircle2, AlertTriangle, Cpu, Activity, BarChart2 } from "lucide-react";

export default function PredictionCard({ versionKey, data, activeTab, onTabChange }) {
  if (!data) return null;

  const isFake = data.prediction === "Fake";
  const statusColor = isFake ? "var(--accent-red)" : "var(--accent-green)";
  const statusBg = isFake ? "rgba(239, 68, 68, 0.12)" : "rgba(16, 185, 129, 0.12)";

  return (
    <div style={{
      background: "var(--bg-card)",
      border: `1px solid ${activeTab === versionKey ? "var(--accent-blue)" : "var(--border-color)"}`,
      borderRadius: "16px",
      padding: "1.5rem",
      backdropFilter: "blur(8px)",
      boxShadow: activeTab === versionKey ? "0 0 20px rgba(59, 130, 246, 0.25)" : "none",
      transition: "all 0.3s ease"
    }}>
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "1rem" }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <h2 style={{ fontSize: "1.25rem", fontWeight: "700" }}>{data.model_name}</h2>
            <span style={{
              background: "rgba(255, 255, 255, 0.08)",
              padding: "0.2rem 0.5rem",
              borderRadius: "4px",
              fontSize: "0.75rem",
              color: "var(--text-muted)"
            }}>
              {versionKey === "v1" ? "Dense Artifact Blocks" : "Depthwise-Separable + Dual Gate"}
            </span>
          </div>
          <div style={{ display: "flex", gap: "1rem", marginTop: "0.4rem", fontSize: "0.8rem", color: "var(--text-muted)" }}>
            <span>Parameters: <strong>{(data.params / 1e6).toFixed(2)}M</strong></span>
            <span>Trained Val Accuracy: <strong>{versionKey === "v1" ? "95.85%" : "93.59%"}</strong></span>
          </div>
        </div>

        <div style={{
          background: statusBg,
          color: statusColor,
          padding: "0.4rem 0.8rem",
          borderRadius: "8px",
          display: "flex",
          alignItems: "center",
          gap: "0.4rem",
          fontWeight: "700",
          fontSize: "1rem",
          border: `1px solid ${statusColor}`
        }}>
          {isFake ? <AlertTriangle size={18} /> : <CheckCircle2 size={18} />}
          {data.prediction.toUpperCase()}
        </div>
      </div>

      {/* Probabilities Meter */}
      <div style={{ marginBottom: "1.5rem" }}>
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.85rem", marginBottom: "0.4rem" }}>
          <span>Classification Confidence</span>
          <strong>{data.confidence_percent}%</strong>
        </div>
        <div style={{
          height: "10px",
          background: "rgba(255, 255, 255, 0.08)",
          borderRadius: "5px",
          overflow: "hidden",
          position: "relative"
        }}>
          <div style={{
            height: "100%",
            width: `${data.fake_probability * 100}%`,
            background: "linear-gradient(90deg, #10b981, #ef4444)",
            borderRadius: "5px",
            transition: "width 0.6s ease"
          }} />
        </div>
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.75rem", color: "var(--text-muted)", marginTop: "0.25rem" }}>
          <span>Real Prob: {(data.real_probability * 100).toFixed(1)}%</span>
          <span>Fake Prob: {(data.fake_probability * 100).toFixed(1)}%</span>
        </div>
      </div>

      {/* Artifact Maps Gallery */}
      <div>
        <h4 style={{ fontSize: "0.9rem", color: "var(--text-muted)", marginBottom: "0.75rem", textTransform: "uppercase", letterSpacing: "0.5px" }}>
          Extracted Artifact Heatmaps
        </h4>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "0.75rem" }}>
          {["block1", "block2", "block3"].map((blk, idx) => {
            const blockData = data.heatmaps[blk];
            return (
              <div key={blk} style={{
                background: "rgba(0, 0, 0, 0.3)",
                border: "1px solid var(--border-color)",
                borderRadius: "8px",
                padding: "0.5rem",
                textAlign: "center"
              }}>
                <img
                  src={blockData.gate}
                  alt={`${blk} Artifact Gate`}
                  style={{ width: "100%", borderRadius: "6px", display: "block" }}
                />
                <div style={{ marginTop: "0.4rem", fontSize: "0.75rem", fontWeight: "600" }}>
                  Stage {idx + 1} Gate
                </div>
                <div style={{ fontSize: "0.65rem", color: "var(--text-muted)" }}>
                  {idx === 0 && "Fine Texture / Boundary"}
                  {idx === 1 && "Mid-Level Semantics"}
                  {idx === 2 && "Global Face Mask"}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
