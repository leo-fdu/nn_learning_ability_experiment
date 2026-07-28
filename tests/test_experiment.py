from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from experiment import (  # noqa: E402
    analytic_bayes,
    apply_standardizer_in_place,
    architecture_widths,
    build_model,
    count_parameters,
    dataset_diagnostics,
    fit_standardizer,
    generate_split,
    load_config,
)


class DataGenerationTests(unittest.TestCase):
    def test_generation_is_exactly_reproducible(self) -> None:
        arguments = {
            "sample_count": 128,
            "max_dimension": 64,
            "informative_dimensions": 4,
            "master_seed": 20260727,
            "repeat": 3,
            "split_name": "train",
        }
        first = generate_split(**arguments)
        second = generate_split(**arguments)
        np.testing.assert_array_equal(first.features, second.features)
        np.testing.assert_array_equal(first.targets, second.targets)

    def test_dimension_conditions_are_prefix_nested(self) -> None:
        split = generate_split(
            sample_count=64,
            max_dimension=4096,
            informative_dimensions=1,
            master_seed=20260727,
            repeat=0,
            split_name="validation",
        )
        np.testing.assert_array_equal(
            split.features[:, :64], split.features[:, :4096][:, :64]
        )

    def test_one_dimensional_population_moments(self) -> None:
        split = generate_split(
            sample_count=100_000,
            max_dimension=8,
            informative_dimensions=1,
            master_seed=20260727,
            repeat=0,
            split_name="train",
        )
        y = split.targets.astype(np.float64)
        x0 = split.features[:, 0].astype(np.float64)
        noise = split.features[:, 1].astype(np.float64)
        self.assertLess(abs(y.mean()), 0.02)
        self.assertLess(abs(y.var() - 1.0), 0.04)
        self.assertLess(abs(x0.var() - 2.0), 0.04)
        self.assertLess(abs(np.cov(x0, y, ddof=0)[0, 1] - 1.0), 0.04)
        self.assertLess(abs(noise.var() - 1.0), 0.04)
        self.assertLess(abs(np.cov(noise, y, ddof=0)[0, 1]), 0.04)

    def test_four_dimensional_population_moments(self) -> None:
        split = generate_split(
            sample_count=100_000,
            max_dimension=8,
            informative_dimensions=4,
            master_seed=20260727,
            repeat=0,
            split_name="train",
        )
        y = split.targets.astype(np.float64)
        signals = split.features[:, :4].astype(np.float64)
        for index in range(4):
            self.assertLess(abs(signals[:, index].var() - 1.125), 0.04)
            self.assertLess(
                abs(np.cov(signals[:, index], y, ddof=0)[0, 1] - 0.25),
                0.04,
            )
        self.assertLess(
            abs(np.cov(signals[:, 0], signals[:, 1], ddof=0)[0, 1] - 0.0625),
            0.04,
        )

    def test_standardizer_uses_supplied_training_statistics(self) -> None:
        train = np.array([[1.0, 10.0], [3.0, 14.0]], dtype=np.float32)
        validation = np.array([[5.0, 18.0]], dtype=np.float32)
        means, standard_deviations = fit_standardizer(train, 1e-8)
        apply_standardizer_in_place(train, means, standard_deviations)
        apply_standardizer_in_place(validation, means, standard_deviations)
        np.testing.assert_allclose(train.mean(axis=0), 0.0, atol=1e-7)
        np.testing.assert_allclose(train.std(axis=0), 1.0, atol=1e-7)
        np.testing.assert_allclose(validation, [[3.0, 3.0]], atol=1e-7)

    def test_diagnostic_schema_is_identical_for_both_signal_modes(self) -> None:
        one = generate_split(
            sample_count=32,
            max_dimension=64,
            informative_dimensions=1,
            master_seed=20260727,
            repeat=0,
            split_name="test",
        )
        four = generate_split(
            sample_count=32,
            max_dimension=64,
            informative_dimensions=4,
            master_seed=20260727,
            repeat=0,
            split_name="test",
        )
        one_diagnostics = dataset_diagnostics(one, 1)
        four_diagnostics = dataset_diagnostics(four, 4)
        self.assertEqual(one_diagnostics.keys(), four_diagnostics.keys())
        self.assertTrue(np.isnan(one_diagnostics["feature_3_mean"]))
        self.assertTrue(np.isfinite(four_diagnostics["feature_3_mean"]))


class ModelAndTheoryTests(unittest.TestCase):
    def test_main_config_matches_the_fixed_contract(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "main.json")
        self.assertEqual(config["dimensions"], [64, 128, 256, 512, 1024, 2048, 4096])
        self.assertEqual(config["repeats"], 10)

    def test_architecture_widths(self) -> None:
        expected = {
            64: [64, 8, 1],
            128: [128, 16, 1],
            256: [256, 32, 1],
            512: [512, 64, 8, 1],
            1024: [1024, 128, 16, 1],
            2048: [2048, 256, 32, 1],
            4096: [4096, 512, 64, 8, 1],
        }
        for dimension, widths in expected.items():
            with self.subTest(dimension=dimension):
                self.assertEqual(architecture_widths(dimension), widths)

    def test_model_layers_and_parameter_counts(self) -> None:
        expected_parameters = {
            64: 529,
            128: 2081,
            256: 8257,
            512: 33361,
            1024: 133281,
            2048: 532801,
            4096: 2131025,
        }
        for dimension, parameter_count in expected_parameters.items():
            with self.subTest(dimension=dimension):
                model = build_model(dimension)
                self.assertIsInstance(model[-1], nn.Linear)
                self.assertEqual(model[-1].out_features, 1)
                self.assertEqual(count_parameters(model), parameter_count)
                for layer in list(model)[:-1:2]:
                    self.assertIsInstance(layer, nn.Linear)

    def test_bayes_references(self) -> None:
        one = analytic_bayes(1)
        four = analytic_bayes(4)
        self.assertAlmostEqual(one["mse"], 0.5)
        self.assertAlmostEqual(one["r2"], 0.5)
        self.assertAlmostEqual(four["mse"], 17.0 / 21.0)
        self.assertAlmostEqual(four["r2"], 4.0 / 21.0)

    def test_forward_shape(self) -> None:
        model = build_model(512)
        result = model(torch.zeros(7, 512))
        self.assertEqual(tuple(result.shape), (7, 1))


if __name__ == "__main__":
    unittest.main()
