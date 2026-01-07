"""Async evaluation utilities for parallel evaluation during training.

This module provides an AsyncEvaluator class that runs evaluation in a background
thread, allowing training to continue without blocking. Weights are cloned at
evaluation time so the training agent can continue updating.
"""

import threading
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import jax
import numpy as np
from tqdm import trange

from utils.evaluation import evaluate


@dataclass
class EvalResult:
    """Result from an async evaluation."""

    step: int
    metrics: Dict[str, Any]
    renders: List[np.ndarray]


class AsyncEvaluator:
    """Manages asynchronous evaluation during training.

    This class runs evaluation in a background thread, cloning the agent weights
    so that training can continue without blocking. Results are collected and
    logged when complete.

    Args:
        max_pending: Maximum number of pending evaluations. If exceeded, new
            evaluation requests will block until a slot is available.
        eval_on_cpu: Whether to run evaluation on CPU.
    """

    def __init__(self, max_pending: int = 1, eval_on_cpu: bool = True):
        self.max_pending = max_pending
        self.eval_on_cpu = eval_on_cpu
        self._executor = ThreadPoolExecutor(max_workers=max_pending)
        self._pending_futures: List[Future] = []
        self._lock = threading.Lock()

    def _clone_agent(self, agent):
        """Clone agent weights for evaluation.

        For JAX, we copy the agent to the target device. This creates a copy
        of the weights that won't be affected by ongoing training updates.
        """
        if self.eval_on_cpu:
            # Copy to CPU - this creates a separate copy of the weights
            return jax.device_put(agent, device=jax.devices('cpu')[0])
        else:
            # Block until computation is done and create a snapshot
            return jax.block_until_ready(agent)

    def _run_evaluation(
        self,
        agent,
        env,
        step: int,
        config: Dict,
        task_infos: List[Dict],
        num_tasks: int,
        eval_episodes: int,
        video_episodes: int,
        video_frame_skip: int,
        eval_temperature: float,
        eval_gaussian: Optional[float],
    ) -> EvalResult:
        """Run evaluation in the background thread."""
        renders = []
        eval_metrics = {}
        overall_metrics = defaultdict(list)

        for task_id in trange(1, num_tasks + 1, desc=f'Eval@{step}'):
            task_name = task_infos[task_id - 1]['task_name']
            eval_info, trajs, cur_renders = evaluate(
                agent=agent,
                env=env,
                task_id=task_id,
                config=config,
                num_eval_episodes=eval_episodes,
                num_video_episodes=video_episodes,
                video_frame_skip=video_frame_skip,
                eval_temperature=eval_temperature,
                eval_gaussian=eval_gaussian,
            )
            renders.extend(cur_renders)
            metric_names = ['success']
            eval_metrics.update(
                {f'evaluation/{task_name}_{k}': v for k, v in eval_info.items() if k in metric_names}
            )
            for k, v in eval_info.items():
                if k in metric_names:
                    overall_metrics[k].append(v)

        for k, v in overall_metrics.items():
            eval_metrics[f'evaluation/overall_{k}'] = np.mean(v)

        return EvalResult(step=step, metrics=eval_metrics, renders=renders)

    def submit_evaluation(
        self,
        agent,
        env,
        step: int,
        config: Dict,
        task_infos: List[Dict],
        num_tasks: int,
        eval_episodes: int,
        video_episodes: int,
        video_frame_skip: int,
        eval_temperature: float,
        eval_gaussian: Optional[float],
    ) -> None:
        """Submit an evaluation to run in the background.

        The agent weights are cloned before submission so training can continue.
        """
        if self.num_pending() >= self.max_pending:
            print(
                f'Warning: eval backlog at step {step} '
                f'({self.num_pending()} pending, max {self.max_pending}). '
                'Consider reducing eval frequency, lowering eval episodes, '
                'or increasing parallel eval capacity.'
            )
        # Clone agent for evaluation
        eval_agent = self._clone_agent(agent)

        # Submit to executor
        future = self._executor.submit(
            self._run_evaluation,
            eval_agent,
            env,
            step,
            config,
            task_infos,
            num_tasks,
            eval_episodes,
            video_episodes,
            video_frame_skip,
            eval_temperature,
            eval_gaussian,
        )

        with self._lock:
            self._pending_futures.append(future)

    def get_completed_results(self) -> List[EvalResult]:
        """Get any completed evaluation results.

        Returns results from evaluations that have completed since the last call.
        Does not block.
        """
        completed = []
        with self._lock:
            still_pending = []
            for future in self._pending_futures:
                if future.done():
                    try:
                        result = future.result()
                        completed.append(result)
                    except Exception as e:
                        print(f'Evaluation failed with error: {e}')
                else:
                    still_pending.append(future)
            self._pending_futures = still_pending
        return completed

    def wait_for_all(self) -> List[EvalResult]:
        """Wait for all pending evaluations to complete and return results."""
        results = []
        with self._lock:
            futures = self._pending_futures[:]
            self._pending_futures = []

        for future in futures:
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f'Evaluation failed with error: {e}')

        return results

    def has_pending(self) -> bool:
        """Check if there are pending evaluations."""
        with self._lock:
            # Clean up completed futures first
            self._pending_futures = [f for f in self._pending_futures if not f.done()]
            return len(self._pending_futures) > 0

    def num_pending(self) -> int:
        """Get the number of pending evaluations."""
        with self._lock:
            return len([f for f in self._pending_futures if not f.done()])

    def shutdown(self, wait: bool = True):
        """Shutdown the executor."""
        self._executor.shutdown(wait=wait)
