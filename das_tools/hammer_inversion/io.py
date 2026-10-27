"""Save and load hammer events (picked dispersion curves) and inversion results.

Both are one compressed .npz each: the arrays under prefixed keys, and
everything else as JSON in a `metadata` entry.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from .data import EventData, PosteriorResult, WaveObservation


def _json_default(obj):
    """JSON form of the NumPy / datetime objects that end up in metadata."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return str(obj)


def save_events(events, filepath):
    """Save a list of EventData to one compressed .npz file."""
    filepath = Path(filepath)
    arrays = {}
    metadata = {"events": []}
    for i, event in enumerate(events):
        event_info = {"event_id": event.event_id, "metadata": event.metadata, "waves": {}}
        for wave_type, wave in event.waves.items():
            prefix = f"event_{i}_{wave_type}"
            arrays[f"{prefix}_frequency"] = wave.frequency
            arrays[f"{prefix}_k_obs"] = wave.k_obs
            arrays[f"{prefix}_sigma_k"] = np.asarray(wave.sigma_k)
            event_info["waves"][wave_type] = {"prefix": prefix, "metadata": wave.metadata}
        metadata["events"].append(event_info)

    arrays["metadata"] = np.array(json.dumps(metadata, default=_json_default))
    np.savez_compressed(filepath, **arrays)
    print(f"Saved {len(events)} events to:")
    print(filepath)


def load_events(filepath):
    """Load the list of EventData written by `save_events`."""
    filepath = Path(filepath)
    events = []
    with np.load(filepath, allow_pickle=False) as data:
        metadata = json.loads(data["metadata"].item())
        for event_info in metadata["events"]:
            waves = {}
            for wave_type, wave_info in event_info["waves"].items():
                prefix = wave_info["prefix"]
                sigma_k = data[f"{prefix}_sigma_k"]
                waves[wave_type] = WaveObservation(
                    wave_type=wave_type,
                    frequency=data[f"{prefix}_frequency"],
                    k_obs=data[f"{prefix}_k_obs"],
                    sigma_k=float(sigma_k) if sigma_k.ndim == 0 else sigma_k,
                    metadata=wave_info["metadata"],
                )
            events.append(EventData(event_id=event_info["event_id"], waves=waves,
                                    metadata=event_info["metadata"]))

    print(f"Loaded {len(events)} events from:")
    print(filepath)
    return events


def save_inversion_results(event_results, global_result, filepath):
    filepath = Path(filepath)
    arrays = {}
    metadata = {"events": {}, "event_ids": [], "global": None}

    def add(prefix, result):
        info = {
            "label": result.label,
            "map_params": result.map_params,
            "map_log_posterior": result.map_log_posterior,
            "metadata": result.metadata,
        }
        if result.log_posterior is not None:
            arrays[f"{prefix}_logp"] = result.log_posterior
        if result.posterior is not None:
            arrays[f"{prefix}_posterior"] = result.posterior
        if result.parameter_axes:
            for name, values in result.parameter_axes.items():
                arrays[f"{prefix}_axis_{name}"] = values
        if result.marginals:
            for name, values in result.marginals.items():
                arrays[f"{prefix}_marginal_{name}"] = values
        return info

    for event_id, result in event_results.items():
        prefix = f"event_{event_id}"
        metadata["event_ids"].append(event_id)
        metadata["events"][str(event_id)] = add(prefix, result)

    metadata["global"] = add("global", global_result)
    arrays["metadata"] = np.array(json.dumps(metadata, default=_json_default))
    np.savez_compressed(filepath, **arrays)


def load_inversion_results(filepath):
    filepath = Path(filepath)
    with np.load(filepath, allow_pickle=False) as data:
        metadata = json.loads(data["metadata"].item())

        def load_one(prefix, info):
            axes = {}
            marginals = {}
            axis_prefix = f"{prefix}_axis_"
            for key in data.files:
                if key.startswith(axis_prefix):
                    name = key[len(axis_prefix):]
                    axes[name] = data[key]
                    mk = f"{prefix}_marginal_{name}"
                    if mk in data:
                        marginals[name] = data[mk]
            return PosteriorResult(
                label=info["label"],
                map_params=info["map_params"],
                map_log_posterior=info["map_log_posterior"],
                log_posterior=data[f"{prefix}_logp"] if f"{prefix}_logp" in data else None,
                posterior=data[f"{prefix}_posterior"] if f"{prefix}_posterior" in data else None,
                parameter_axes=axes or None,
                marginals=marginals or None,
                metadata=info.get("metadata", {}),
            )

        events = {}
        for event_id in metadata["event_ids"]:
            events[event_id] = load_one(
                f"event_{event_id}",
                metadata["events"][str(event_id)],
            )

        global_result = load_one("global", metadata["global"])

    return events, global_result
