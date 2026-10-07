import unittest

import torch

from networks.tsr import MaskAwareTSRAdapter


class MaskAwareTSRAdapterTest(unittest.TestCase):
    def test_zero_initialization_is_exact(self):
        adapter = MaskAwareTSRAdapter(channels=8, hidden_channels=4)
        features = torch.randn(2, 8, 8, 8)
        masks = torch.randint(0, 2, (2, 1, 32, 32)).float()

        output = adapter(features, masks)

        self.assertTrue(torch.equal(output, torch.zeros_like(output)))

    def test_residual_is_restricted_to_hole(self):
        adapter = MaskAwareTSRAdapter(channels=8, hidden_channels=4)
        adapter.project_out.bias.data.fill_(1.0)
        features = torch.randn(1, 8, 8, 8)
        masks = torch.zeros(1, 1, 32, 32)
        masks[:, :, 8:24, 8:24] = 1

        output = adapter(features, masks)
        hole = torch.nn.functional.interpolate(masks, size=(8, 8), mode="nearest")

        self.assertTrue(torch.equal(output * (1 - hole), torch.zeros_like(output)))
        expected = torch.tanh(torch.ones_like(output)) * hole
        self.assertTrue(torch.equal(output * hole, expected))


if __name__ == "__main__":
    unittest.main()
