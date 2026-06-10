from __future__ import annotations

from dataclasses import asdict
from typing import Any

from src_v2.config import ExperimentConfig


def init_wandb_run(config: ExperimentConfig, job_type: str):
    if not config.wandb.enabled:
        return None

    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "wandb is enabled but the package is not installed. Install it with `pip install wandb`."
        ) from exc

    run_name = config.wandb.name.format(job_type=job_type) if config.wandb.name else None
    tags = list(config.wandb.tags)
    if job_type not in tags:
        tags.append(job_type)

    return wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        name=run_name,
        group=config.wandb.group,
        tags=tags,
        job_type=job_type,
        mode=config.wandb.mode,
        config=asdict(config),
    )


def log_wandb_metrics(run, metrics: dict[str, Any], step: int | None = None) -> None:
    if run is None:
        return
    run.log(metrics, step=step)


def update_wandb_summary(run, **items: Any) -> None:
    if run is None:
        return
    for key, value in items.items():
        run.summary[key] = value


def finish_wandb_run(run) -> None:
    if run is None:
        return
    run.finish()
