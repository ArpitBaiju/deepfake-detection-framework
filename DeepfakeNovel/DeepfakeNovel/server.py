"""
FastAPI Server for DAF-Net Live Deepfake Detection.
Serves predictions and real-time artifact heatmap extractions for both DAF-Net V1 and DAF-Net V2.
"""
import io
import os
import base64
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.models.daf_net import DAFNetV1, DAFNetV2

app = FastAPI(title="DAF-Net Deepfake Detection API")

# Enable CORS for React frontend (Vite default: http://localhost:5173, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[API] Initializing models on {DEVICE}...")

# Load models
models = {}

def load_model(version_name, model_fn, ckpt_name):
    ckpt_path = os.path.join("checkpoints", ckpt_name)
    model = model_fn().to(DEVICE)
    if os.path.exists(ckpt_path):
        print(f"[API] Loading {version_name} from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        print(f"[API] {version_name} successfully loaded.")
    else:
        print(f"[API WARNING] Checkpoint {ckpt_path} not found for {version_name}.")
    model.eval()
    return model

models["v1"] = load_model("DAF-Net V1", DAFNetV1, "dafnet_v1_best.pth")
models["v2"] = load_model("DAF-Net V2", DAFNetV2, "dafnet_v2_best.pth")

# Preprocessing transforms (128x128 match trained input)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def tensor_to_base64_heatmap(tensor_2d, colormap='inferno'):
    """Converts 2D tensor heatmap into base64 PNG image"""
    fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
    ax.imshow(tensor_2d, cmap=colormap)
    ax.axis("off")
    plt.tight_layout(pad=0)
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")

@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "device": str(DEVICE),
        "available_models": list(models.keys()),
        "v1_params": sum(p.numel() for p in models["v1"].parameters()),
        "v2_params": sum(p.numel() for p in models["v2"].parameters())
    }

@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    model_version: str = Form("both")  # "v1", "v2", or "both"
):
    try:
        image_bytes = await file.read()
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    # Preprocess
    input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

    results = {}
    target_models = ["v1", "v2"] if model_version == "both" else [model_version.lower()]

    for v in target_models:
        if v not in models:
            continue
        model = models[v]
        
        with torch.no_grad():
            logit, art_maps = model(input_tensor, return_artifact_maps=True)
            prob = torch.sigmoid(logit).item()

        is_fake = prob >= 0.5
        confidence = prob if is_fake else (1.0 - prob)

        # Generate Base64 Heatmaps for Block 1, 2, 3
        heatmaps = {}
        for blk in ["block1", "block2", "block3"]:
            gate_tensor = art_maps[blk]["gate"].squeeze().mean(dim=0).cpu().numpy()
            dev_tensor = art_maps[blk]["deviation"].squeeze().mean(dim=0).cpu().numpy()
            heatmaps[blk] = {
                "gate": tensor_to_base64_heatmap(gate_tensor, colormap="inferno" if v == "v1" else "viridis"),
                "deviation": tensor_to_base64_heatmap(dev_tensor, colormap="magma"),
                "alpha": art_maps[blk].get("alpha", 1.0)
            }

        results[v] = {
            "model_name": f"DAF-Net {v.upper()}",
            "params": sum(p.numel() for p in model.parameters()),
            "prediction": "Fake" if is_fake else "Real",
            "fake_probability": round(prob, 4),
            "real_probability": round(1.0 - prob, 4),
            "confidence_percent": round(confidence * 100, 2),
            "heatmaps": heatmaps
        }

    return {
        "filename": file.filename,
        "results": results
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
