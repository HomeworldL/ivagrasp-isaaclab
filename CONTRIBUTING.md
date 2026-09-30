# Contributing

Issues and pull requests are welcome. For a behavior change, briefly describe the task and expected result in an issue or pull request before making a large change.

Use the [recorded environment](ENVIRONMENT.md) for simulator work. Keep task configuration, asset conversion, and training scripts separate, and avoid committing datasets, checkpoints, logs, or machine-specific paths. If a hand asset changes, describe its source, license, and conversion steps in the pull request.

Run the relevant tests before opening a pull request. The lightweight CI check is:

```bash
python -m compileall -q source scripts tests
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=source python -m pytest -q \
  tests/test_object_assignment.py tests/test_mdp_core.py tests/test_moving_speed_curriculum.py
```

Tests that require Isaac Sim or object datasets should be run locally with those dependencies available; include the command and outcome in the pull request. Keep pull requests focused and explain any behavior or compatibility change.
