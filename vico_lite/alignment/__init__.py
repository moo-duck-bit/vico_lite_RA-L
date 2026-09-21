"""Optional learned-alignment extension point.

ViCo-Lite is training-free: no projection head is instantiated in any reported
experiment. ``CrossModalAligner`` is kept only so a learned alignment variant
can be dropped in without restructuring the perception pipeline.
"""

from .align import CrossModalAligner

__all__ = ["CrossModalAligner"]
