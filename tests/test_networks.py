"""Network compatibility checks; runnable with Python's built-in unittest."""

import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

CODE_DIR = Path(__file__).resolve().parents[1] / "Code"
sys.path.insert(0, str(CODE_DIR))

try:
    import torch
    import networks
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is required for network tests")
class NetworkCompatibilityTests(unittest.TestCase):
    def check_networks(self, device):
        configs = json.loads((CODE_DIR / "optimized_ppo_configs.json").read_text())
        original_builder = networks._build_recurrent_layer

        def strict_builder(recurrent_type, input_size, hidden_size):
            # Reproduce the newer PyTorch check even when running older PyTorch.
            self.assertIs(type(input_size), int)
            return original_builder(recurrent_type, input_size, hidden_size)

        for key, entry in configs.items():
            with self.subTest(configuration=key, device=device):
                config = entry["ppo"]
                args = (config["num_conv_layers"], config["num_filters"], config["strides"],
                        config["kernels_size"], config["num_mlp_layers"],
                        config["recurrent_units"], config["mlp_neurons"])
                with patch.object(networks, "_build_recurrent_layer", side_effect=strict_builder):
                    actor = networks.ModularActor(*args, action_dim=8,
                                                  recurrent_type=config["recurrent_type"]).to(device)
                    critic = networks.ModularCritic(*args,
                                                    recurrent_type=config["recurrent_type"]).to(device)
                self.assertIs(type(actor.out_features), int)
                self.assertIs(type(critic.out_features), int)
                self.assertEqual(actor.out_features, critic.out_features)
                state = torch.randn(2, 3, 48, 46, device=device)
                hidden = actor.initial_hidden(2, device)
                probabilities, _ = actor.pi(state, hidden)
                values = critic(state, hidden)
                self.assertEqual(tuple(probabilities.shape), (2, 1, 8))
                self.assertEqual(tuple(values.shape), (2, 1, 1))
                self.assertTrue(torch.isfinite(probabilities).all().item())
                self.assertTrue(torch.isfinite(values).all().item())
                torch.testing.assert_close(probabilities.sum(-1), torch.ones(2, 1, device=device))
                loss = -probabilities[..., 0].log().mean() + values.square().mean()
                loss.backward()
                self.assertIsNotNone(actor.recurrent.weight_ih_l0.grad)
                self.assertIsNotNone(critic.recurrent.weight_ih_l0.grad)

    def test_selected_networks_cpu(self):
        self.check_networks("cpu")

    def test_selected_networks_cuda(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        self.check_networks("cuda")


if __name__ == "__main__":
    unittest.main()
