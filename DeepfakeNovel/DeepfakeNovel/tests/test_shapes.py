"""
Comprehensive Sanity Test Suite for all DAF-Net Versions (V1, V2, V3, V4) and Baselines.
Verifies forward pass, backward gradient propagation, intermediate artifact shape consistency,
and parameter counts.
"""
import torch
import unittest
from src.models.blocks import (
    ArtifactBlockV1,
    ArtifactBlockV2,
    ArtifactBlockV3,
    ArtifactBlockV4
)
from src.models.daf_net import (
    DAFNet,
    DAFNetV1,
    DAFNetV2,
    DAFNetV3,
    DAFNetV4,
    VanillaBaselineCNN
)
from src.utils.metrics import profile_model


class TestAllDAFNetVersions(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.channels = 64
        self.spatial = 32

    def test_all_individual_blocks(self):
        blocks = {
            "V1": ArtifactBlockV1(self.channels),
            "V2": ArtifactBlockV2(self.channels),
            "V3": ArtifactBlockV3(self.channels),
            "V4": ArtifactBlockV4(self.channels)
        }
        
        for v_name, blk in blocks.items():
            x = torch.randn(self.batch_size, self.channels, self.spatial, self.spatial, requires_grad=True)
            out, art_info = blk(x, return_artifacts=True)
            self.assertEqual(out.shape, (self.batch_size, self.channels, self.spatial, self.spatial), f"Mismatch in {v_name} out shape")
            self.assertEqual(art_info["gate"].shape, (self.batch_size, self.channels, self.spatial, self.spatial), f"Mismatch in {v_name} gate shape")
            
            # Test gradient propagation
            loss = out.sum()
            loss.backward()
            self.assertIsNotNone(x.grad, f"Gradient missing on input for block {v_name}")

    def test_full_model_versions(self):
        versions = ["v1", "v2", "v3", "v4"]
        x = torch.randn(self.batch_size, 3, 128, 128)
        
        print("\n--- Model Profiling Summary (Input: 128x128) ---")
        vanilla = VanillaBaselineCNN()
        vanilla_stats = profile_model(vanilla, input_size=(1, 3, 128, 128))
        print(f"Vanilla CNN: {vanilla_stats['total_parameters']:,} params | Latency: {vanilla_stats['inference_latency_ms']} ms")
        
        for v in versions:
            model = DAFNet(version=v)
            
            # Forward pass
            logits, art_maps = model(x, return_artifact_maps=True)
            self.assertEqual(logits.shape, (self.batch_size, 1), f"Mismatch in {v} logits")
            self.assertIn("block1", art_maps)
            self.assertIn("block2", art_maps)
            self.assertIn("block3", art_maps)
            
            # Backward pass
            loss = logits.sum()
            loss.backward()
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.assertIsNotNone(param.grad, f"Gradient missing for {v} layer: {name}")
                    
            stats = profile_model(model, input_size=(1, 3, 128, 128))
            print(f"DAF-Net {v.upper()}: {stats['total_parameters']:,} params | Latency: {stats['inference_latency_ms']} ms")


if __name__ == "__main__":
    unittest.main()
