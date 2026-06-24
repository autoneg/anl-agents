from __future__ import annotations

try:
    from .hybrid_agent import HybridNegotiator
except ImportError:
    from hybrid_agent import HybridNegotiator


class MyNegotiator(HybridNegotiator):
    """
    Submission entrypoint.

    This class intentionally aliases the strongest current hybrid policy so the
    template CLI and submission packaging keep working without extra changes.
    """

    pass


class AnlOmegaNegotiator(HybridNegotiator):
    """
    Final submission entrypoint.
    """

    pass
