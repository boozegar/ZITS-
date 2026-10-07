import copy
import unittest

import torch

from networks.generators import (
    DenseGuidanceDecoderAdapter,
    SpatialZeroRAAdapter,
    StructureEncoder,
    _apply_dense_decoder_adapter,
    _configure_dense_decoder_adapters,
    _select_dense_decoder_guidance,
)


def structure_config():
    return {
        "prior_ch": 5,
        "rezero_for_mpe": True,
        "rel_pos_num": 128,
    }


class SpatialZeroRATest(unittest.TestCase):
    def test_dense_decoder_channel_and_stage_ablation_preserves_interface(self):
        module = torch.nn.Module()
        _configure_dense_decoder_adapters(
            module,
            {
                "dense_decoder_zero_ra": {
                    "enabled": True,
                    "guidance_channels": 4,
                    "hidden_channels": 8,
                    "guidance_names": ["edge"],
                    "enabled_stages": [2],
                }
            },
        )
        batch = {
            name: torch.full((1, 1, 8, 8), value)
            for value, name in enumerate(
                ("edge", "gradientx", "gradienty", "line"),
                start=1,
            )
        }
        guidance = _select_dense_decoder_guidance(module, batch)
        self.assertEqual(guidance.shape, (1, 4, 8, 8))
        self.assertTrue(torch.equal(guidance[:, :1], batch["edge"]))
        self.assertEqual(torch.count_nonzero(guidance[:, 1:]), 0)

        feature = torch.randn(1, 256, 4, 4)
        mask = torch.ones(1, 1, 8, 8)
        self.assertIs(
            _apply_dense_decoder_adapter(module, 0, feature, guidance, mask),
            feature,
        )

    def test_dense_decoder_adapter_is_zero_initialized_and_hole_gated(self):
        adapter = DenseGuidanceDecoderAdapter(4, 16, hidden_channels=8)
        guidance = torch.randn(2, 4, 32, 32)
        mask = torch.zeros(2, 1, 32, 32)
        mask[:, :, 8:24, 8:24] = 1

        self.assertEqual(torch.count_nonzero(adapter(guidance, mask, (16, 16))), 0)

        with torch.no_grad():
            adapter.project_out.bias.fill_(0.25)
        delta = adapter(guidance, mask, (16, 16))
        self.assertEqual(torch.count_nonzero(delta[:, :, :4, :]), 0)
        self.assertGreater(torch.count_nonzero(delta[:, :, 4:12, 4:12]), 0)

    def test_dense_decoder_adapter_receives_guidance_gradient(self):
        adapter = DenseGuidanceDecoderAdapter(4, 8, hidden_channels=8)
        with torch.no_grad():
            adapter.project_out.weight.fill_(0.1)
        guidance = torch.randn(1, 4, 16, 16, requires_grad=True)
        mask = torch.ones(1, 1, 16, 16)

        adapter(guidance, mask, (8, 8)).sum().backward()

        self.assertIsNotNone(guidance.grad)
        self.assertGreater(guidance.grad.abs().sum(), 0)

    def test_feature_conditioned_decoder_adapter_starts_from_guidance_only(self):
        adapter = DenseGuidanceDecoderAdapter(
            4,
            8,
            hidden_channels=8,
            feature_channels=8,
        )
        guidance = torch.randn(1, 4, 16, 16)
        mask = torch.ones(1, 1, 16, 16)
        feature = torch.randn(1, 8, 8, 8)
        with torch.no_grad():
            adapter.project_out.weight.fill_(0.1)

        initial = adapter(guidance, mask, (8, 8), feature=feature)
        guidance_only = adapter(guidance, mask, (8, 8), feature=torch.zeros_like(feature))
        self.assertTrue(torch.equal(initial, guidance_only))

        with torch.no_grad():
            adapter.feature_project.weight.fill_(0.1)
        conditioned = adapter(guidance, mask, (8, 8), feature=feature)
        self.assertFalse(torch.equal(conditioned, guidance_only))

    def test_spatial_adapter_is_zero_initialized_and_hole_gated(self):
        adapter = SpatialZeroRAAdapter(16)
        feature = torch.randn(2, 16, 16, 16)
        mask = torch.zeros(2, 1, 16, 16)
        mask[:, :, 4:12, 4:12] = 1

        self.assertEqual(torch.count_nonzero(adapter(feature, mask)), 0)

        with torch.no_grad():
            adapter.project_out.bias.fill_(0.25)
        delta = adapter(feature, mask)
        self.assertEqual(torch.count_nonzero(delta[:, :, :4, :]), 0)
        self.assertGreater(torch.count_nonzero(delta[:, :, 4:12, 4:12]), 0)

    def test_sa_zero_ra_matches_original_encoder_at_initialization(self):
        torch.manual_seed(7)
        baseline = StructureEncoder(structure_config()).eval()
        sa_config = copy.deepcopy(structure_config())
        sa_config["spatial_zero_ra"] = {"enabled": True}
        treatment = StructureEncoder(sa_config).eval()
        incompatible = treatment.load_state_dict(baseline.state_dict(), strict=False)

        self.assertFalse(incompatible.unexpected_keys)
        self.assertTrue(incompatible.missing_keys)
        self.assertTrue(
            all(key.startswith("spatial_adapters.") for key in incompatible.missing_keys)
        )

        structure = torch.randn(2, 5, 64, 64)
        mask = torch.randint(0, 2, (2, 1, 64, 64), dtype=torch.float32)
        rel_pos = torch.zeros(2, 64, 64, dtype=torch.long)
        direct = torch.zeros(2, 64, 64, 4)

        with torch.no_grad():
            baseline_outputs = baseline(structure, rel_pos, direct)
            treatment_outputs = treatment(structure, rel_pos, direct, mask=mask)

        for baseline_output, treatment_output in zip(baseline_outputs, treatment_outputs):
            if isinstance(baseline_output, list):
                for baseline_feature, treatment_feature in zip(baseline_output, treatment_output):
                    self.assertTrue(torch.equal(baseline_feature, treatment_feature))
            else:
                self.assertTrue(torch.equal(baseline_output, treatment_output))

    def test_channel_residual_matches_original_encoder_at_initialization(self):
        torch.manual_seed(11)
        baseline = StructureEncoder(structure_config()).eval()
        sa_config = copy.deepcopy(structure_config())
        sa_config["spatial_zero_ra"] = {
            "enabled": True,
            "mode": "channel_residual",
        }
        treatment = StructureEncoder(sa_config).eval()
        treatment.load_state_dict(baseline.state_dict(), strict=False)

        structure = torch.randn(1, 5, 64, 64)
        mask = torch.randint(0, 2, (1, 1, 64, 64), dtype=torch.float32)
        rel_pos = torch.zeros(1, 64, 64, dtype=torch.long)
        direct = torch.zeros(1, 64, 64, 4)
        with torch.no_grad():
            baseline_outputs = baseline(structure, rel_pos, direct)
            treatment_outputs = treatment(structure, rel_pos, direct, mask=mask)

        for baseline_feature, treatment_feature in zip(
            baseline_outputs[0], treatment_outputs[0]
        ):
            self.assertTrue(torch.equal(baseline_feature, treatment_feature))


if __name__ == "__main__":
    unittest.main()
