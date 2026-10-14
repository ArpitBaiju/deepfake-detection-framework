import React from "react";
import { Zap, Activity, Shield, Check, Award } from "lucide-react";

export default function ComparisonTable() {
  const comparisonData = [
    {
      metric: "Architectural Focus",
      v1: "Dense |3x3 - 5x5| + Scalar Gate",
      v2: "Depthwise-Separable + Dual Spatial/Channel Gate",
      impact: "High parameter efficiency"
    },
    {
      metric: "Total Parameters",
      v1: "3,402,852 (~3.40M)",
      v2: "622,231 (~0.62M)",
      impact: "81.7% parameter reduction (5.5x smaller)"
    },
    {
      metric: "Training Epoch Speed",
      v1: "~13.5 mins / epoch",
      v2: "~8.1 mins / epoch",
      impact: "40% faster computation"
    },
    {
      metric: "Validation Accuracy",
      v1: "95.85%",
      v2: "93.59%",
      impact: "V1 slightly stronger on subtle textures"
    },
    {
      metric: "Validation F1-Score",
      v1: "0.9582",
      v2: "0.9363",
      impact: "Both show robust balanced precision/recall"
    },
    {
      metric: "ROC-AUC Score",
      v1: "0.9928",
      v2: "0.9836",
      impact: "Both exceed 0.98 AUC separation"
    },
    {
      metric: "False Positive Rate (FPR)",
      v1: "3.75% (743 / 19,787)",
      v2: "7.36% (1,458 / 19,787)",
      impact: "V1 has lower false alarm rate"
    },
    {
      metric: "Ideal Deployment Target",
      v1: "Cloud / High-Precision Moderation",
      v2: "Edge Devices / Mobile / Webcam Feeds",
      impact: "Clear operational distinction"
    }
  ];

  return (
    <div style={{
      background: "var(--bg-card)",
      border: "1px solid var(--border-color)",
      borderRadius: "16px",
      padding: "1.5rem",
      backdropFilter: "blur(8px)",
      marginTop: "2rem"
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "1rem" }}>
        <Award size={22} color="#fbbf24" />
        <h3 style={{ fontSize: "1.2rem", fontWeight: "700" }}>
          Empirical Architecture Trade-off (Review & Viva Presentation Guide)
        </h3>
      </div>
      <p style={{ fontSize: "0.85rem", color: "var(--text-muted)", marginBottom: "1.25rem" }}>
        Controlled evaluation under identical 12-epoch training protocols on the 140,000-image dataset.
      </p>

      <div style={{ overflowX: "auto" }}>
        <table style={{
          width: "100%",
          borderCollapse: "collapse",
          fontSize: "0.85rem",
          textAlign: "left"
        }}>
          <thead>
            <tr style={{ borderBottom: "1px solid var(--border-color)", color: "var(--text-muted)" }}>
              <th style={{ padding: "0.75rem 1rem" }}>Evaluation Dimension</th>
              <th style={{ padding: "0.75rem 1rem", color: "#60a5fa" }}>DAF-Net V1</th>
              <th style={{ padding: "0.75rem 1rem", color: "#a78bfa" }}>DAF-Net V2</th>
              <th style={{ padding: "0.75rem 1rem" }}>Key Takeaway for Panel</th>
            </tr>
          </thead>
          <tbody>
            {comparisonData.map((row, i) => (
              <tr key={i} style={{
                borderBottom: "1px solid rgba(255, 255, 255, 0.04)",
                background: i % 2 === 0 ? "rgba(255, 255, 255, 0.02)" : "transparent"
              }}>
                <td style={{ padding: "0.75rem 1rem", fontWeight: "600" }}>{row.metric}</td>
                <td style={{ padding: "0.75rem 1rem", color: "#93c5fd" }}>{row.v1}</td>
                <td style={{ padding: "0.75rem 1rem", color: "#c4b5fd" }}>{row.v2}</td>
                <td style={{ padding: "0.75rem 1rem", color: "var(--text-muted)" }}>{row.impact}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
