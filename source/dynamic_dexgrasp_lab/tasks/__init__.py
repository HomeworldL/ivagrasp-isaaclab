"""Task registrations for the repository."""

__all__ = ["register_tasks"]


def register_tasks() -> None:
    """Import task variants for gym registration side effects."""
    from .dexgrasp_float import register_tasks as _register_dexgrasp_float_tasks
    from .dexgrasp_moving import register_tasks as _register_dexgrasp_moving_tasks

    _register_dexgrasp_float_tasks()
    _register_dexgrasp_moving_tasks()
