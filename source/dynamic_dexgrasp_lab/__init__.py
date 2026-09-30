"""Custom IsaacLab task package for dynamic dexterous grasp migration."""

__all__ = ["register_tasks"]


def register_tasks():
    """Import task modules for gym registration side effects."""
    from .tasks import register_tasks as _register_tasks

    _register_tasks()
