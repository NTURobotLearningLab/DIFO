"""Command-line scripts."""

# Add our custom environments to Gym registry.

import difo  # noqa: F401
import difo.baselines  # noqa: F401
import envs  # noqa: F401

try:
    # pytype: disable=import-error
    import seals  # noqa: F401

    # pytype: enable=import-error
except ImportError as e:
    print(f"Failed to import: {e}")
