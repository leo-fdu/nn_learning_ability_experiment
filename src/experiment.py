from __future__ import annotations

import copy
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SPLIT_IDS = {"train": 0, "validation": 1, "test": 2}
COMPONENT_IDS = {"target": 0, "epsilon": 1, "eta": 2, "noise": 3}


@dataclass
class DatasetSplit:
    features: np.ndarray
    targets: np.ndarray


@dataclass
class DatasetBundle:
    train: DatasetSplit
    validation: DatasetSplit
    test: DatasetSplit
    train_mean: np.ndarray
    train_std: np.ndarray


@dataclass
class TrainingResult:
    best_epoch: int
    epochs_ran: int
    best_train_mse: float
    best_validation_mse: float
    history: list[dict[str, float | int]]
    elapsed_seconds: float


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    dimensions = list(config["dimensions"])
    informative = list(config["informative_dimensions"])
    split_sizes = config["split_sizes"]
    data = config["data"]
    model = config["model"]
    training = config["training"]

    expected_dimensions = [64, 128, 256, 512, 1024, 2048, 4096]
    if dimensions != expected_dimensions:
        raise ValueError(
            f"this experiment requires dimensions={expected_dimensions}"
        )
    if informative != [1, 4]:
        raise ValueError("this experiment requires informative_dimensions=[1, 4]")
    if int(config["master_seed"]) != 20260727:
        raise ValueError("this experiment requires master_seed=20260727")
    if int(config["repeats"]) != 10:
        raise ValueError("this experiment requires repeats=10")
    expected_split_sizes = {"train": 32768, "validation": 4096, "test": 8192}
    if dict(split_sizes) != expected_split_sizes:
        raise ValueError(
            f"this experiment requires split_sizes={expected_split_sizes}"
        )
    if data["dtype"] != "float32":
        raise ValueError("the experiment contract requires float32 arrays")
    if not bool(data["standardize_inputs"]):
        raise ValueError("the experiment contract requires input standardization")
    if float(data["standard_deviation_floor"]) != 1e-8:
        raise ValueError(
            "the experiment contract requires standard_deviation_floor=1e-8"
        )
    if int(model["compression_factor"]) != 8:
        raise ValueError("the experiment contract requires compression_factor=8")
    if int(model["scalar_output_threshold"]) != 32:
        raise ValueError("the experiment contract requires scalar_output_threshold=32")
    if model["activation"].lower() != "relu":
        raise ValueError("the experiment contract requires ReLU hidden activations")
    if bool(model["batch_normalization"]) or float(model["dropout"]) != 0.0:
        raise ValueError("BatchNorm and Dropout are excluded by the experiment contract")
    if not bool(model["use_bias"]):
        raise ValueError("the experiment contract requires Linear layer biases")
    if training["optimizer"].lower() != "adamw":
        raise ValueError("the experiment contract requires AdamW")
    expected_training_values = {
        "batch_size": 256,
        "evaluation_batch_size": 1024,
        "maximum_epochs": 200,
        "learning_rate": 1e-3,
        "epsilon": 1e-8,
        "weight_decay": 1e-4,
    }
    for key, expected in expected_training_values.items():
        if training[key] != expected:
            raise ValueError(
                f"the experiment contract requires training.{key}={expected}"
            )
    if list(training["betas"]) != [0.9, 0.999]:
        raise ValueError("the experiment contract requires AdamW betas=[0.9, 0.999]")
    if int(training["data_loader_workers"]) != 0:
        raise ValueError("data_loader_workers must remain 0 for reproducibility")
    if bool(training["automatic_mixed_precision"]):
        raise ValueError("automatic mixed precision is excluded by the experiment contract")


def derive_seed(master_seed: int, *parts: int) -> int:
    sequence = np.random.SeedSequence([int(master_seed), *(int(part) for part in parts)])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _make_rng(master_seed: int, *parts: int) -> np.random.Generator:
    sequence = np.random.SeedSequence([int(master_seed), *(int(part) for part in parts)])
    return np.random.Generator(np.random.PCG64(sequence))


def generate_split(
    *,
    sample_count: int,
    max_dimension: int,
    informative_dimensions: int,
    master_seed: int,
    repeat: int,
    split_name: str,
) -> DatasetSplit:
    if split_name not in SPLIT_IDS:
        raise ValueError(f"unknown split name: {split_name}")
    if informative_dimensions not in {1, 4}:
        raise ValueError("informative_dimensions must be 1 or 4")
    if max_dimension <= informative_dimensions:
        raise ValueError("max_dimension must exceed informative_dimensions")

    split_id = SPLIT_IDS[split_name]
    target_rng = _make_rng(
        master_seed,
        informative_dimensions,
        repeat,
        split_id,
        COMPONENT_IDS["target"],
    )
    noise_rng = _make_rng(
        master_seed,
        informative_dimensions,
        repeat,
        split_id,
        COMPONENT_IDS["noise"],
    )

    targets = target_rng.standard_normal(sample_count, dtype=np.float32)
    features = np.empty((sample_count, max_dimension), dtype=np.float32)
    noise_rng.standard_normal(features.shape, dtype=np.float32, out=features)

    epsilon_rng = _make_rng(
        master_seed,
        informative_dimensions,
        repeat,
        split_id,
        COMPONENT_IDS["epsilon"],
    )
    if informative_dimensions == 1:
        epsilon = epsilon_rng.standard_normal(sample_count, dtype=np.float32)
        features[:, 0] = targets + epsilon
    else:
        eta_rng = _make_rng(
            master_seed,
            informative_dimensions,
            repeat,
            split_id,
            COMPONENT_IDS["eta"],
        )
        epsilon = epsilon_rng.standard_normal(
            (sample_count, informative_dimensions), dtype=np.float32
        )
        eta = eta_rng.standard_normal(
            (sample_count, informative_dimensions), dtype=np.float32
        )
        features[:, :informative_dimensions] = (
            (targets[:, None] + epsilon) / 4.0 + eta
        )

    return DatasetSplit(features=features, targets=targets)


def fit_standardizer(
    features: np.ndarray, standard_deviation_floor: float
) -> tuple[np.ndarray, np.ndarray]:
    means = features.mean(axis=0, dtype=np.float64)
    standard_deviations = features.std(axis=0, dtype=np.float64, ddof=0)
    invalid = np.flatnonzero(standard_deviations < standard_deviation_floor)
    if invalid.size:
        preview = invalid[:10].tolist()
        raise ValueError(
            f"{invalid.size} columns have standard deviation below "
            f"{standard_deviation_floor}; first indices: {preview}"
        )
    return means, standard_deviations


def apply_standardizer_in_place(
    features: np.ndarray, means: np.ndarray, standard_deviations: np.ndarray
) -> None:
    if features.shape[1] != means.shape[0] or means.shape != standard_deviations.shape:
        raise ValueError("standardization statistics do not match feature columns")
    features -= means.astype(np.float32, copy=False)
    features /= standard_deviations.astype(np.float32, copy=False)


def generate_dataset_bundle(
    *,
    split_sizes: Mapping[str, int],
    max_dimension: int,
    informative_dimensions: int,
    master_seed: int,
    repeat: int,
    standardize_inputs: bool,
    standard_deviation_floor: float,
) -> DatasetBundle:
    train = generate_split(
        sample_count=int(split_sizes["train"]),
        max_dimension=max_dimension,
        informative_dimensions=informative_dimensions,
        master_seed=master_seed,
        repeat=repeat,
        split_name="train",
    )
    means, standard_deviations = fit_standardizer(
        train.features, standard_deviation_floor
    )

    validation = generate_split(
        sample_count=int(split_sizes["validation"]),
        max_dimension=max_dimension,
        informative_dimensions=informative_dimensions,
        master_seed=master_seed,
        repeat=repeat,
        split_name="validation",
    )
    test = generate_split(
        sample_count=int(split_sizes["test"]),
        max_dimension=max_dimension,
        informative_dimensions=informative_dimensions,
        master_seed=master_seed,
        repeat=repeat,
        split_name="test",
    )

    if standardize_inputs:
        for split in (train, validation, test):
            apply_standardizer_in_place(
                split.features, means, standard_deviations
            )

    return DatasetBundle(
        train=train,
        validation=validation,
        test=test,
        train_mean=means,
        train_std=standard_deviations,
    )


def architecture_widths(
    input_dimension: int, compression_factor: int = 8, scalar_threshold: int = 32
) -> list[int]:
    if input_dimension <= 0:
        raise ValueError("input_dimension must be positive")
    widths = [int(input_dimension)]
    current = int(input_dimension)
    while current > scalar_threshold:
        if current % compression_factor != 0:
            raise ValueError(
                f"dimension {current} is not divisible by compression factor "
                f"{compression_factor}"
            )
        current //= compression_factor
        widths.append(current)
    widths.append(1)
    return widths


def build_model(
    input_dimension: int, compression_factor: int = 8, scalar_threshold: int = 32
) -> nn.Sequential:
    widths = architecture_widths(
        input_dimension, compression_factor, scalar_threshold
    )
    layers: list[nn.Module] = []
    linear_layers: list[nn.Linear] = []
    for index, (input_width, output_width) in enumerate(
        zip(widths[:-1], widths[1:])
    ):
        linear = nn.Linear(input_width, output_width, bias=True)
        layers.append(linear)
        linear_layers.append(linear)
        if index < len(widths) - 2:
            layers.append(nn.ReLU())

    for hidden_layer in linear_layers[:-1]:
        nn.init.kaiming_uniform_(
            hidden_layer.weight, a=0.0, mode="fan_in", nonlinearity="relu"
        )
        if hidden_layer.bias is not None:
            nn.init.zeros_(hidden_layer.bias)
    return nn.Sequential(*layers)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def analytic_bayes(informative_dimensions: int) -> dict[str, float]:
    if informative_dimensions == 1:
        return {"mse": 0.5, "r2": 0.5}
    if informative_dimensions == 4:
        return {"mse": 17.0 / 21.0, "r2": 4.0 / 21.0}
    raise ValueError("informative_dimensions must be 1 or 4")


def set_model_seed(
    *,
    master_seed: int,
    informative_dimensions: int,
    input_dimension: int,
    repeat: int,
    deterministic_algorithms: bool,
    deterministic_warn_only: bool,
) -> int:
    model_seed = derive_seed(
        master_seed, 100, informative_dimensions, input_dimension, repeat
    )
    random.seed(model_seed)
    np.random.seed(model_seed)
    torch.manual_seed(model_seed)
    if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(model_seed)
    torch.use_deterministic_algorithms(
        deterministic_algorithms, warn_only=deterministic_warn_only
    )
    return model_seed


def resolve_device(requested_device: str) -> torch.device:
    requested = requested_device.lower()
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS was requested but torch.backends.mps.is_available() is False"
            )
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError("device must be explicitly set to 'mps' or 'cpu'")


def _make_loader(
    split: DatasetSplit,
    input_dimension: int,
    batch_size: int,
    shuffle: bool,
    generator_seed: int | None,
) -> DataLoader:
    features = torch.from_numpy(split.features[:, :input_dimension])
    targets = torch.from_numpy(split.targets[:, None])
    generator = None
    if generator_seed is not None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(generator_seed)
    return DataLoader(
        TensorDataset(features, targets),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
        drop_last=False,
        pin_memory=False,
    )


def _mse_on_loader(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> float:
    model.eval()
    squared_error_sum = 0.0
    sample_count = 0
    with torch.no_grad():
        for features, targets in loader:
            predictions = model(features.to(device))
            targets_device = targets.to(device)
            squared_error_sum += torch.sum(
                (predictions - targets_device) ** 2
            ).item()
            sample_count += targets.shape[0]
    if sample_count == 0:
        raise ValueError("cannot evaluate an empty loader")
    return squared_error_sum / sample_count


def train_model(
    *,
    model: nn.Module,
    train_split: DatasetSplit,
    validation_split: DatasetSplit,
    input_dimension: int,
    device: torch.device,
    model_seed: int,
    training_config: Mapping[str, Any],
) -> TrainingResult:
    batch_size = int(training_config["batch_size"])
    evaluation_batch_size = int(training_config["evaluation_batch_size"])
    train_loader = _make_loader(
        train_split,
        input_dimension,
        batch_size,
        shuffle=True,
        generator_seed=derive_seed(model_seed, 1),
    )
    validation_loader = _make_loader(
        validation_split,
        input_dimension,
        evaluation_batch_size,
        shuffle=False,
        generator_seed=None,
    )

    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        betas=tuple(float(value) for value in training_config["betas"]),
        eps=float(training_config["epsilon"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    scheduler_config = training_config["scheduler"]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(scheduler_config["factor"]),
        patience=int(scheduler_config["patience"]),
        threshold=float(scheduler_config["threshold"]),
        min_lr=float(scheduler_config["minimum_learning_rate"]),
    )

    early_stopping = training_config["early_stopping"]
    patience = int(early_stopping["patience"])
    minimum_delta = float(early_stopping["minimum_delta"])
    maximum_epochs = int(training_config["maximum_epochs"])

    best_validation_mse = math.inf
    best_train_mse = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []
    started_at = time.perf_counter()

    for epoch in range(1, maximum_epochs + 1):
        epoch_started_at = time.perf_counter()
        model.train()
        train_squared_error_sum = 0.0
        train_sample_count = 0

        for features, targets in train_loader:
            features_device = features.to(device)
            targets_device = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(features_device)
            loss = torch.mean((predictions - targets_device) ** 2)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite training loss at epoch {epoch}"
                )
            loss.backward()
            optimizer.step()
            train_squared_error_sum += loss.item() * targets.shape[0]
            train_sample_count += targets.shape[0]

        train_mse = train_squared_error_sum / train_sample_count
        validation_mse = _mse_on_loader(model, validation_loader, device)
        if not math.isfinite(validation_mse):
            raise FloatingPointError(
                f"non-finite validation loss at epoch {epoch}"
            )
        learning_rate = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "epoch": epoch,
                "learning_rate": learning_rate,
                "train_mse": train_mse,
                "validation_mse": validation_mse,
                "epoch_seconds": time.perf_counter() - epoch_started_at,
            }
        )

        if validation_mse < best_validation_mse - minimum_delta:
            best_validation_mse = validation_mse
            best_train_mse = train_mse
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        scheduler.step(validation_mse)
        if epochs_without_improvement >= patience:
            break

    if best_state is None:
        raise RuntimeError("training completed without a finite best model")
    model.load_state_dict(best_state)
    elapsed_seconds = time.perf_counter() - started_at
    return TrainingResult(
        best_epoch=best_epoch,
        epochs_ran=len(history),
        best_train_mse=best_train_mse,
        best_validation_mse=best_validation_mse,
        history=history,
        elapsed_seconds=elapsed_seconds,
    )


def evaluate_model(
    *,
    model: nn.Module,
    split: DatasetSplit,
    input_dimension: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    loader = _make_loader(
        split,
        input_dimension,
        batch_size,
        shuffle=False,
        generator_seed=None,
    )
    model.eval()
    predictions_all: list[np.ndarray] = []
    targets_all: list[np.ndarray] = []
    with torch.no_grad():
        for features, targets in loader:
            predictions = model(features.to(device)).cpu().numpy().reshape(-1)
            predictions_all.append(predictions)
            targets_all.append(targets.numpy().reshape(-1))

    predictions = np.concatenate(predictions_all).astype(np.float64)
    targets = np.concatenate(targets_all).astype(np.float64)
    residuals = predictions - targets
    mse = float(np.mean(residuals**2))
    mae = float(np.mean(np.abs(residuals)))
    denominator = float(np.sum((targets - targets.mean()) ** 2))
    if denominator <= 0.0:
        raise ValueError("R2 is undefined because all test targets are identical")
    r2 = 1.0 - float(np.sum(residuals**2)) / denominator
    return {"mse": mse, "mae": mae, "r2": r2}


def dataset_diagnostics(
    split: DatasetSplit, informative_dimensions: int
) -> dict[str, float]:
    targets = split.targets.astype(np.float64, copy=False)
    raw = {
        "target_mean": float(targets.mean()),
        "target_variance": float(targets.var(ddof=0)),
    }
    for index in range(4):
        if index < informative_dimensions:
            feature = split.features[:, index].astype(np.float64, copy=False)
            raw[f"feature_{index}_mean"] = float(feature.mean())
            raw[f"feature_{index}_variance"] = float(feature.var(ddof=0))
            raw[f"feature_{index}_target_covariance"] = float(
                np.mean((feature - feature.mean()) * (targets - targets.mean()))
            )
        else:
            raw[f"feature_{index}_mean"] = math.nan
            raw[f"feature_{index}_variance"] = math.nan
            raw[f"feature_{index}_target_covariance"] = math.nan
    return raw


def environment_snapshot(device: torch.device) -> dict[str, Any]:
    versions: dict[str, str] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    try:
        import matplotlib
        import scipy
        import sklearn

        versions.update(
            {
                "matplotlib": matplotlib.__version__,
                "scipy": scipy.__version__,
                "scikit_learn": sklearn.__version__,
            }
        )
    except ImportError:
        pass
    return {
        "versions": versions,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "requested_and_resolved_device": str(device),
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "command": sys.argv,
    }


def smoke_config(config: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(config))
    result["dimensions"] = [64, 4096]
    result["informative_dimensions"] = [1, 4]
    result["repeats"] = 1
    result["split_sizes"] = {"train": 512, "validation": 128, "test": 256}
    result["training"]["batch_size"] = 128
    result["training"]["evaluation_batch_size"] = 256
    result["training"]["maximum_epochs"] = 2
    result["training"]["early_stopping"]["patience"] = 2
    return result


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def append_csv_rows(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows_list = list(rows)
    if not rows_list:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows_list)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)
