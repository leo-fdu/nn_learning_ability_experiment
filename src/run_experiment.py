from __future__ import annotations

import argparse
import gc
import json
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiment import (
    analytic_bayes,
    append_csv_rows,
    build_model,
    count_parameters,
    dataset_diagnostics,
    derive_seed,
    environment_snapshot,
    evaluate_model,
    generate_dataset_bundle,
    load_config,
    resolve_device,
    set_model_seed,
    smoke_config,
    train_model,
    write_json,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the high-dimensional low-signal MLP experiment."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cpu"), required=True)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run the fixed four-model smoke configuration.",
    )
    return parser.parse_args()


def prepare_output_directory(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {path}. "
            "Choose a new directory; existing results will not be overwritten."
        )
    path.mkdir(parents=True, exist_ok=True)
    (path / "checkpoints").mkdir(exist_ok=True)
    (path / "standardization").mkdir(exist_ok=True)


def run() -> None:
    arguments = parse_arguments()
    config = load_config(arguments.config.resolve())
    if arguments.smoke_test:
        config = smoke_config(config)

    output_directory = arguments.output_dir.resolve()
    prepare_output_directory(output_directory)
    device = resolve_device(arguments.device)
    write_json(output_directory / "config_resolved.json", config)
    write_json(output_directory / "environment.json", environment_snapshot(device))

    dimensions = [int(value) for value in config["dimensions"]]
    informative_values = [
        int(value) for value in config["informative_dimensions"]
    ]
    repeat_count = int(config["repeats"])
    master_seed = int(config["master_seed"])
    training_config = config["training"]
    data_config = config["data"]
    save_checkpoints = bool(config["analysis"]["save_checkpoints"])

    for informative_dimensions in informative_values:
        bayes = analytic_bayes(informative_dimensions)
        for repeat in range(repeat_count):
            print(
                f"[data] n={informative_dimensions} repeat={repeat} "
                f"max_dimension={max(dimensions)}",
                flush=True,
            )
            bundle = generate_dataset_bundle(
                split_sizes=config["split_sizes"],
                max_dimension=max(dimensions),
                informative_dimensions=informative_dimensions,
                master_seed=master_seed,
                repeat=repeat,
                standardize_inputs=bool(data_config["standardize_inputs"]),
                standard_deviation_floor=float(
                    data_config["standard_deviation_floor"]
                ),
            )
            np.savez_compressed(
                output_directory
                / "standardization"
                / f"n{informative_dimensions}_r{repeat}.npz",
                mean=bundle.train_mean,
                std=bundle.train_std,
            )
            diagnostics_rows: list[dict[str, Any]] = []
            for split_name, split in (
                ("train", bundle.train),
                ("validation", bundle.validation),
                ("test", bundle.test),
            ):
                diagnostics_rows.append(
                    {
                        "n_informative": informative_dimensions,
                        "repeat": repeat,
                        "split": split_name,
                        **dataset_diagnostics(split, informative_dimensions),
                    }
                )
            append_csv_rows(
                output_directory / "data_diagnostics.csv", diagnostics_rows
            )

            for dimension in dimensions:
                run_identifier = (
                    f"n{informative_dimensions}_d{dimension}_r{repeat}"
                )
                print(f"[train] {run_identifier}", flush=True)
                model_seed = set_model_seed(
                    master_seed=master_seed,
                    informative_dimensions=informative_dimensions,
                    input_dimension=dimension,
                    repeat=repeat,
                    deterministic_algorithms=bool(
                        training_config["deterministic_algorithms"]
                    ),
                    deterministic_warn_only=bool(
                        training_config["deterministic_warn_only"]
                    ),
                )
                model = build_model(
                    dimension,
                    compression_factor=int(config["model"]["compression_factor"]),
                    scalar_threshold=int(
                        config["model"]["scalar_output_threshold"]
                    ),
                )
                parameter_count = count_parameters(model)
                try:
                    training_result = train_model(
                        model=model,
                        train_split=bundle.train,
                        validation_split=bundle.validation,
                        input_dimension=dimension,
                        device=device,
                        model_seed=model_seed,
                        training_config=training_config,
                    )
                    test_metrics = evaluate_model(
                        model=model,
                        split=bundle.test,
                        input_dimension=dimension,
                        batch_size=int(
                            training_config["evaluation_batch_size"]
                        ),
                        device=device,
                    )
                    checkpoint_path = ""
                    if save_checkpoints:
                        checkpoint = (
                            output_directory
                            / "checkpoints"
                            / f"{run_identifier}.pt"
                        )
                        torch.save(
                            {
                                "run_identifier": run_identifier,
                                "state_dict": {
                                    key: value.detach().cpu()
                                    for key, value in model.state_dict().items()
                                },
                                "input_dimension": dimension,
                                "n_informative": informative_dimensions,
                                "repeat": repeat,
                                "model_seed": model_seed,
                                "config": config,
                            },
                            checkpoint,
                        )
                        checkpoint_path = str(checkpoint)

                    history_rows = [
                        {
                            "run_identifier": run_identifier,
                            "n_informative": informative_dimensions,
                            "total_dim": dimension,
                            "repeat": repeat,
                            **history_row,
                        }
                        for history_row in training_result.history
                    ]
                    append_csv_rows(
                        output_directory / "history.csv", history_rows
                    )
                    run_row = {
                        "run_identifier": run_identifier,
                        "n_informative": informative_dimensions,
                        "total_dim": dimension,
                        "repeat": repeat,
                        "data_seed": derive_seed(
                            master_seed, informative_dimensions, repeat
                        ),
                        "model_seed": model_seed,
                        "parameter_count": parameter_count,
                        "best_epoch": training_result.best_epoch,
                        "epochs_ran": training_result.epochs_ran,
                        "best_train_mse": training_result.best_train_mse,
                        "best_val_mse": training_result.best_validation_mse,
                        "test_mse": test_metrics["mse"],
                        "test_mae": test_metrics["mae"],
                        "test_r2": test_metrics["r2"],
                        "bayes_mse": bayes["mse"],
                        "bayes_r2": bayes["r2"],
                        "capture_ratio": test_metrics["r2"] / bayes["r2"],
                        "excess_mse": test_metrics["mse"] - bayes["mse"],
                        "train_seconds": training_result.elapsed_seconds,
                        "device": str(device),
                        "status": "success",
                        "checkpoint_path": checkpoint_path,
                    }
                    append_csv_rows(output_directory / "runs.csv", [run_row])
                    print(
                        f"[done] {run_identifier} "
                        f"R2={test_metrics['r2']:.6f} "
                        f"MSE={test_metrics['mse']:.6f} "
                        f"epochs={training_result.epochs_ran}",
                        flush=True,
                    )
                except Exception as error:
                    failure = {
                        "run_identifier": run_identifier,
                        "n_informative": informative_dimensions,
                        "total_dim": dimension,
                        "repeat": repeat,
                        "model_seed": model_seed,
                        "status": "failed",
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                        "traceback": traceback.format_exc(),
                    }
                    with (output_directory / "failures.jsonl").open(
                        "a", encoding="utf-8"
                    ) as handle:
                        handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
                    raise
                finally:
                    del model
                    if device.type == "mps":
                        torch.mps.empty_cache()

            del bundle
            gc.collect()
            if device.type == "mps":
                torch.mps.empty_cache()

    print(f"[complete] results written to {output_directory}", flush=True)


if __name__ == "__main__":
    run()
