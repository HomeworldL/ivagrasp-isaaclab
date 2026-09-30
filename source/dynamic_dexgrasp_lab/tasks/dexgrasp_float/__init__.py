"""Floating-hand dexterous grasp task package."""

__all__ = ["register_tasks"]


def register_tasks() -> None:
    """Import concrete task variants for gym registration side effects."""
    from . import config  # noqa: F401
