"""Train the CPU shared 3D residual baseline before full PINN optimization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from gbm_pinn.shared_forecaster import (
    SharedResidualForecaster,
    dice_score,
    load_transition_manifest,
    prepare_transition,
    uniform_sample_indices,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-index", type=Path, required=True)
    parser.add_argument("--model-selection-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--samples-per-transition", type=int, default=4096)
    parser.add_argument("--downsample", type=int, default=4)
    parser.add_argument("--seed", type=int, default=162)
    args = parser.parse_args()
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    training_records = load_transition_manifest(args.training_index, required_role="training")
    selection_records = load_transition_manifest(
        args.model_selection_index, required_role="model_selection"
    )
    print(f"Preparing {len(training_records)} training transitions", flush=True)
    train_x, train_p, train_y, training_patient_ids = _prepare_training_samples(
        training_records,
        args.samples_per_transition,
        args.downsample,
        rng,
    )
    print(f"Preparing {len(selection_records)} model-selection transitions", flush=True)
    selection = [
        prepare_transition(value, downsample=args.downsample) for value in selection_records
    ]
    select_x, select_p, select_y = _sample(selection, args.samples_per_transition, rng)
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-6] = 1.0
    train_x = (train_x - mean) / scale
    select_x = (select_x - mean) / scale
    model = SharedResidualForecaster(train_x.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    tx, tp, ty = map(torch.from_numpy, (train_x, train_p, train_y))
    vx, vp, vy = map(torch.from_numpy, (select_x, select_p, select_y))
    best_loss = float("inf")
    best_state = None
    stale = 0
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(tx.shape[0])
        for start in range(0, tx.shape[0], 8192):
            batch = order[start : start + 8192]
            optimizer.zero_grad()
            logits = model(tx[batch], tp[batch]).squeeze(1)
            correction = model.network(tx[batch]).squeeze(1)
            loss = criterion(logits, ty[batch]) + 0.02 * torch.mean(correction.square())
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(criterion(model(vx, vp).squeeze(1), vy))
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"epoch={epoch + 1} selection_bce={validation_loss:.6f}", flush=True)
        if stale >= 20:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    candidates = []
    for correction_scale in (0.0, 0.25, 0.5, 0.75, 1.0):
        for threshold in np.arange(0.25, 0.76, 0.05):
            candidate_metrics = _evaluate(
                model,
                selection,
                mean,
                scale,
                float(threshold),
                correction_scale,
            )
            candidates.append((candidate_metrics["mean_dice"], correction_scale, threshold))
    _, correction_scale, threshold_value = max(candidates)
    correction_scale = float(correction_scale)
    threshold = float(threshold_value)
    metrics = _evaluate(model, selection, mean, scale, threshold, correction_scale)
    artifact = {
        "model_type": "shared_3d_residual_training_gate",
        "not_final_pinn": True,
        "seed": args.seed,
        "downsample": args.downsample,
        "training_transition_count": len(training_records),
        "training_patient_count": len(training_patient_ids),
        "model_selection_transition_count": len(selection),
        "model_selection_patient_count": len({item.patient_id for item in selection}),
        "best_selection_bce": best_loss,
        "selected_threshold": threshold,
        "selected_correction_scale": correction_scale,
        "model_selection_metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_mean": mean,
            "feature_scale": scale,
            "metadata": artifact,
        },
        args.output,
    )
    args.output.with_suffix(".json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(artifact, indent=2, sort_keys=True))


def _sample(transitions, count, rng):
    features, persistence, target = [], [], []
    for transition in transitions:
        indices = uniform_sample_indices(transition, count, rng)
        features.append(transition.features[indices])
        persistence.append(transition.persistence[indices])
        target.append(transition.target[indices])
    return (
        np.concatenate(features).astype(np.float32),
        np.concatenate(persistence).astype(np.float32),
        np.concatenate(target).astype(np.float32),
    )


def _prepare_training_samples(records, count, downsample, rng):
    features, persistence, target = [], [], []
    patient_ids = set()
    for index, record in enumerate(records, start=1):
        transition = prepare_transition(record, downsample=downsample)
        indices = uniform_sample_indices(transition, count, rng)
        features.append(transition.features[indices])
        persistence.append(transition.persistence[indices])
        target.append(transition.target[indices])
        patient_ids.add(transition.patient_id)
        if index % 10 == 0 or index == len(records):
            print(f"prepared_training={index}/{len(records)}", flush=True)
    return (
        np.concatenate(features).astype(np.float32),
        np.concatenate(persistence).astype(np.float32),
        np.concatenate(target).astype(np.float32),
        patient_ids,
    )


def _evaluate(model, transitions, mean, scale, threshold, correction_scale):
    records = []
    model.eval()
    with torch.no_grad():
        for transition in transitions:
            probabilities = []
            for start in range(0, transition.target.size, 65536):
                features = (transition.features[start : start + 65536] - mean) / scale
                persistence = transition.persistence[start : start + 65536]
                feature_tensor = torch.from_numpy(features)
                persistence_tensor = torch.from_numpy(persistence)
                prior = torch.where(
                    persistence_tensor.reshape(-1, 1) > 0.5,
                    torch.full_like(persistence_tensor.reshape(-1, 1), 3.0),
                    torch.full_like(persistence_tensor.reshape(-1, 1), -3.0),
                )
                logits = prior + correction_scale * model.network(feature_tensor)
                probabilities.append(torch.sigmoid(logits).squeeze(1).numpy())
            prediction = np.concatenate(probabilities) >= threshold
            forecast_dice = dice_score(prediction, transition.target)
            persistence_dice = dice_score(transition.persistence, transition.target)
            records.append(
                {
                    "transition_id": transition.transition_id,
                    "patient_id": transition.patient_id,
                    "forecast_dice": forecast_dice,
                    "persistence_dice": persistence_dice,
                    "dice_skill_over_persistence": forecast_dice - persistence_dice,
                }
            )
    skills = [record["dice_skill_over_persistence"] for record in records]
    return {
        "mean_dice": float(np.mean([record["forecast_dice"] for record in records])),
        "median_dice": float(np.median([record["forecast_dice"] for record in records])),
        "median_persistence_dice": float(
            np.median([record["persistence_dice"] for record in records])
        ),
        "median_dice_skill_over_persistence": float(np.median(skills)),
        "mean_dice_skill_over_persistence": float(np.mean(skills)),
        "n_beating_persistence": sum(value > 0 for value in skills),
        "n_transitions": len(records),
        "records": records,
    }


if __name__ == "__main__":
    main()
