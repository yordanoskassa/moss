"""Programmatic ClawEval runner for MOSS integration."""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from claweval.config import EvalConfig, ModelConfig, Settings as ClawSettings, load_config
from claweval.reporter import aggregate_results, save_json_results
from claweval.runner import TaskResult, run_tasks
from claweval.task_loader import load_tasks

from moss.eval.claweval_bridge import ClawEvalBridge


class ClawEvalRunner:
    """Run ClawEval tasks programmatically and bridge results into MOSS."""

    def __init__(
        self,
        config_path: Path | str | None = None,
        model_id: str | None = None,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        categories: list[str] | None = None,
        scoring_mode: str = "deterministic",
        timeout: int = 300,
    ) -> None:
        self._categories = categories
        self._scoring_mode = scoring_mode

        if config_path is not None:
            self._config = load_config(config_path)
        else:
            # Build a config programmatically.
            resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "")
            if not resolved_key:
                raise ValueError(
                    "No API key provided. Set OPENAI_API_KEY or pass api_key=."
                )
            model_name = model_id or "gpt-4o-mini"
            self._config = EvalConfig(
                models=[
                    ModelConfig(
                        id=model_name,
                        name=model_name,
                        provider="openai",
                        base_url=base_url,
                        api_key=resolved_key,
                    )
                ],
                settings=ClawSettings(
                    scoring_mode=scoring_mode,
                    timeout_seconds=timeout,
                    warmup_requests=0,
                    runs_per_task=1,
                    parallel_tasks=1,
                    categories=categories
                    or [
                        "tool_calling",
                        "coding",
                        "reasoning",
                        "writing",
                        "research",
                        "memory",
                        "speed",
                    ],
                    raw={},
                ),
            )

        self._results: list[TaskResult] = []
        self._bridge: ClawEvalBridge | None = None

    def run(
        self,
        categories: list[str] | None = None,
        model_id: str | None = None,
    ) -> list[TaskResult]:
        """Execute ClawEval tasks and return raw results.

        Args:
            categories: Override category filter for this run.
            model_id: Run only a specific model from the config.

        Returns:
            List of ``TaskResult`` objects from claweval.
        """
        cats = categories or self._categories
        tasks = load_tasks(categories=cats)

        models = self._config.models
        if model_id:
            models = [m for m in models if m.id == model_id]
            if not models:
                raise ValueError(f"Model '{model_id}' not found in config")

        all_results: list[TaskResult] = []
        for model in models:
            results = run_tasks(
                tasks=tasks,
                model=model,
                settings=self._config.settings,
                scoring_mode=self._scoring_mode,
            )
            all_results.extend(results)

        self._results = all_results
        self._bridge = None  # Reset bridge cache.
        return all_results

    def save_results(self, output_dir: Path | str = Path("results")) -> Path:
        """Save results to JSON and return the file path."""
        if not self._results:
            raise RuntimeError("No results to save. Call run() first.")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_names = {m.id: m.name for m in self._config.models}
        return save_json_results(
            self._results,
            model_names=model_names,
            output_dir=output_dir,
        )

    def bridge(self) -> ClawEvalBridge:
        """Get a ClawEvalBridge populated with the current results.

        Converts raw ``TaskResult`` objects into the JSON structure that
        ``ClawEvalBridge`` expects, without writing to disk.
        """
        if self._bridge is not None:
            return self._bridge

        if not self._results:
            raise RuntimeError("No results available. Call run() first.")

        # Build the same structure that save_json_results produces.
        summaries = aggregate_results(
            self._results,
            model_names={m.id: m.name for m in self._config.models},
        )

        models_dict: dict[str, Any] = {}
        for model_id, summary in summaries.items():
            task_dicts = []
            for r in self._results:
                if r.model_id != model_id:
                    continue
                task_dicts.append({
                    "task_id": r.task_id,
                    "model_id": r.model_id,
                    "score": asdict(r.score) if r.score else None,
                    "timing": asdict(r.timing),
                    "response_text": r.response_text,
                    "tool_calls_made": r.tool_calls_made,
                    "error": r.error,
                })
            models_dict[model_id] = {
                "name": summary.name,
                "overall": summary.overall,
                "categories": summary.categories,
                "tasks": task_dicts,
            }

        bridge = ClawEvalBridge()
        bridge.load_from_dict({"models": models_dict})
        self._bridge = bridge
        return bridge
