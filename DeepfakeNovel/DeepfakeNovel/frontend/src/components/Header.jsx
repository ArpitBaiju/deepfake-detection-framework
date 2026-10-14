import React from "react";
import { ShieldCheck, Cpu, Layers, Sparkles } from "lucide-react";

export default function Header({ systemStatus }) {
  return (
    <header style={{
      borderBottom: "1px solid var(--border-color)",
      background: "rgba(10, 14, 23, 0.8)",
      backdropFilter: "blur(12px)",
      padding: "1rem 2rem",
      display: "flex",
      justifyContent: "space-between",
      alignItems: "center",
      position: "sticky",
      top: 0,
      zIndex: 100
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
        <div style={{
          width: "44px",
          height: "44px",
          borderRadius: "10px",
          background: "linear-gradient(135deg, #3b82f6, #8b5cf6)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          boxShadow: "0 0 15px rgba(59, 130, 246, 0.4)"
        }}>
          <ShieldCheck size={26} color="#fff" />
        </div>
        <div>
          <h1 style={{
            fontSize: "1.4rem",
            fontWeight: "700",
            letterSpacing: "-0.5px",
            background: "linear-gradient(90deg, #60a5fa, #c084fc)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent"
          }}>
            DAF-Net Deepfake Inspection Studio
          </h1>
          <p style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
            Novel Progressive Artifact-Focusing Network Evaluation & Diagnostics
          </p>
        </div>
      </div>

      <div style={{ display: "flex", gap: "1rem", alignItems: "center" }}>
        <div style={{
          display: "flex",
          alignItems: "center",
          gap: "0.5rem",
          background: "rgba(255, 255, 255, 0.05)",
          padding: "0.4rem 0.8rem",
          borderRadius: "8px",
          fontSize: "0.8rem",
          border: "1px solid var(--border-color)"
        }}>
          <Cpu size={15} color="#60a5fa" />
          <span>Device: <strong>{systemStatus?.device?.toUpperCase() || "CPU"}</strong></span>
        </div>

        <div style={{
          display: "flex",
          alignItems: "center",
          gap: "0.5rem",
          background: systemStatus?.status === "healthy" ? "rgba(16, 185, 129, 0.1)" : "rgba(239, 68, 68, 0.1)",
          color: systemStatus?.status === "healthy" ? "#10b981" : "#ef4444",
          padding: "0.4rem 0.8rem",
          borderRadius: "8px",
          fontSize: "0.8rem",
          border: "1px solid currentColor"
        }}>
          <span style={{
            width: "8px",
            height: "8px",
            borderRadius: "50%",
            backgroundColor: systemStatus?.status === "healthy" ? "#10b981" : "#ef4444"
          }}></span>
          <span>{systemStatus?.status === "healthy" ? "Both Models Loaded (V1 & V2)" : "API Disconnected"}</span>
        </div>
      </div>
    </header>
  );
}
