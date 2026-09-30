"""Reset-aware cache helpers for dexgrasp task-side features."""

from __future__ import annotations


def _env_cache_token(env) -> tuple[int, int]:
    """返回当前环境步缓存 token。

    中文说明：
    - 仅靠 ``common_step_counter`` 不足以区分 reset 前后的首帧；
    - auto-reset 会发生在同一个 env step 内，这时 step 不变；
    - 因此缓存必须同时带上 ``reset_epoch``，只要 reset 过一次，token 就会变化。
    """

    step = int(getattr(env, "common_step_counter", -1))
    reset_epoch = int(getattr(env, "dexgrasp_float_reset_epoch", 0))
    return step, reset_epoch


def _bump_reset_epoch(env) -> int:
    """在 reset 事件末尾推进一代缓存 epoch。

    中文说明：
    - 这个值不代表物理时间；
    - 它只表示“当前 episode/reset 语义已经切换到下一代”；
    - 任何按 step 缓存的观测/几何特征，只要看到 epoch 变化，就必须重算。
    """

    next_epoch = int(getattr(env, "dexgrasp_float_reset_epoch", 0)) + 1
    env.dexgrasp_float_reset_epoch = next_epoch
    return next_epoch
