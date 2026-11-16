"""evla -- end-to-end Vision-Language-Action lab.

The repository is organised as a pipeline, and each stage only depends on the
interfaces in :mod:`evla.types`::

    sim env / demos  ->  fine-tune  ->  optimize  ->  deploy  ->  evaluate
      evla.envs        lerobot-train  evla.optimize  evla.deploy  evla.eval

Import the heavy stages lazily; ``import evla`` must stay cheap so that the CLI and
the test-suite work on a CPU-only machine without PyTorch installed.
"""

__version__ = "0.1.0"

from evla.types import Observation, Policy, StepResult  # noqa: F401

__all__ = ["Observation", "Policy", "StepResult", "__version__"]
