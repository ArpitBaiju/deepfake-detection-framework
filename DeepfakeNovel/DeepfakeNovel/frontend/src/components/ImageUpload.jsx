import React, { useRef } from "react";
import { UploadCloud, Image as ImageIcon, Sparkles } from "lucide-react";

export default function ImageUpload({ onImageSelected, previewUrl, loading }) {
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      onImageSelected(file);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onImageSelected(e.dataTransfer.files[0]);
    }
  };

  return (
    <div style={{
      background: "var(--bg-card)",
      border: "1px solid var(--border-color)",
      borderRadius: "16px",
      padding: "1.5rem",
      backdropFilter: "blur(8px)"
    }}>
      <h3 style={{ fontSize: "1.1rem", marginBottom: "0.5rem", display: "flex", alignItems: "center", gap: "0.5rem" }}>
        <ImageIcon size={18} color="#60a5fa" />
        Input Face Image
      </h3>
      <p style={{ fontSize: "0.85rem", color: "var(--text-muted)", marginBottom: "1rem" }}>
        Upload any suspect or genuine face image to run live inference across both DAF-Net architectures.
      </p>

      <div
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        style={{
          border: "2px dashed rgba(59, 130, 246, 0.4)",
          borderRadius: "12px",
          padding: previewUrl ? "0.5rem" : "3rem 1.5rem",
          textAlign: "center",
          cursor: "pointer",
          background: "rgba(0, 0, 0, 0.2)",
          position: "relative",
          overflow: "hidden",
          transition: "all 0.2s ease"
        }}
      >
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileChange}
          accept="image/*"
          style={{ display: "none" }}
        />

        {previewUrl ? (
          <div style={{ position: "relative" }}>
            <img
              src={previewUrl}
              alt="Uploaded facial input"
              style={{
                maxWidth: "100%",
                maxHeight: "320px",
                borderRadius: "8px",
                display: "block",
                margin: "0 auto",
                objectFit: "contain"
              }}
            />
            {loading && <div className="scanner-line"></div>}
            <div style={{
              marginTop: "0.5rem",
              fontSize: "0.75rem",
              color: "var(--text-muted)"
            }}>
              Click or drag another image to replace
            </div>
          </div>
        ) : (
          <div>
            <UploadCloud size={48} color="#60a5fa" style={{ marginBottom: "1rem", opacity: 0.8 }} />
            <p style={{ fontWeight: "600", fontSize: "0.95rem", marginBottom: "0.25rem" }}>
              Click to browse or drop facial image here
            </p>
            <p style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
              Supports JPG, JPEG, PNG (Automatically normalized to 128x128 for DAF-Net)
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
