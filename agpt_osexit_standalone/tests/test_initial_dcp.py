import os
import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch import nn

from torchtitan.components.checkpoint import CheckpointManager, ModelWrapper
from torchtitan.models.agpt.config_registry import agpt_2b_yarn


class InitialDCPTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        dcp.save(
            {
                "norm.weight": torch.arange(4, dtype=torch.float32),
                "optimizer": {"sentinel": torch.tensor(42)},
                "train_state": {"step": 49900},
            },
            checkpoint_id=self.source,
        )

    def manager(self, width=4):
        model = nn.Module()
        model.add_module("norm", nn.LayerNorm(width, bias=False))
        manager = CheckpointManager.__new__(CheckpointManager)
        manager.enable = True
        manager.stager = None
        manager.folder = str(self.root / "new_checkpoints")
        manager.initial_model_path = str(self.source)
        manager.states = {
            "model": ModelWrapper(model),
            "train_state": {"step": 0},
            "optimizer": {"sentinel": torch.tensor(-1)},
        }
        manager.exclude_from_loading = []
        return manager, model

    def test_loads_model_without_restoring_training_state(self):
        manager, model = self.manager()
        self.assertTrue(manager.load())
        torch.testing.assert_close(model.norm.weight, torch.arange(4.0))
        self.assertEqual(manager.states["train_state"]["step"], 0)
        self.assertEqual(manager.states["optimizer"]["sentinel"].item(), -1)
        self.assertFalse(Path(manager.folder).exists())

    def test_empty_output_folder_still_loads_source(self):
        manager, model = self.manager()
        Path(manager.folder).mkdir()
        self.assertTrue(manager.load())
        torch.testing.assert_close(model.norm.weight, torch.arange(4.0))

    def test_rejects_wrong_tensor_shapes(self):
        manager, _ = self.manager(width=5)
        with self.assertRaisesRegex(ValueError, "keys/shapes"):
            manager.load()

    def test_rejects_missing_source(self):
        manager, _ = self.manager()
        manager.initial_model_path = str(self.root / "missing")
        with self.assertRaises(FileNotFoundError):
            manager.load()

    def test_local_checkpoint_takes_precedence(self):
        manager, model = self.manager()
        local = Path(manager.folder) / "step-0"
        dcp.save({"norm.weight": torch.full((4,), 9.0)}, checkpoint_id=local)
        self.assertTrue(manager.load())
        torch.testing.assert_close(model.norm.weight, torch.full((4,), 9.0))

    def test_requested_source_matches_all_model_keys(self):
        config = agpt_2b_yarn()
        with torch.device("meta"):
            model = config.model_spec.model.build()
        metadata = dcp.FileSystemReader(os.environ["AGPT_INITIAL_DCP"]).read_metadata()
        state = model.state_dict()
        self.assertEqual(len(state), 111)
        for key, tensor in state.items():
            self.assertIn(key, metadata.state_dict_metadata)
            self.assertEqual(tensor.shape, metadata.state_dict_metadata[key].size)

    def test_actual_requested_dcp_tensor_is_loaded(self):
        source = os.environ["AGPT_INITIAL_DCP"]
        expected = {"norm.weight": torch.zeros(2048)}
        dcp.load(expected, checkpoint_id=source)
        manager, model = self.manager(width=2048)
        manager.initial_model_path = source
        with torch.no_grad():
            model.norm.weight.zero_()
        self.assertTrue(manager.load())
        torch.testing.assert_close(model.norm.weight, expected["norm.weight"])
        self.assertTrue(torch.isfinite(model.norm.weight).all().item())
        self.assertGreater(torch.count_nonzero(model.norm.weight).item(), 0)


if __name__ == "__main__":
    unittest.main()
