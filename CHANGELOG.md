# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/);
version numbers correspond to releases published on [PyPI](https://pypi.org/project/anl-agents/).

## [Unreleased]

- Added ANL 2026 agent submissions and qualification tracking.
- Flagged the 17 ANL 2026 finalists (`get_agents(2026, finalists_only=True)`).
- Fixed a malformed single-line `requirements.txt` for team_20963.
- Relaxed the numpy version constraint and dropped a dead `numpy.lib.introspect` import.
- Reorganized project configuration and build tooling metadata.
- Added GitHub Actions workflows for testing (incl. against negmas's GitHub `main`) and PyPI publishing.

## [0.1.2] - 2025-08-29

- Added ANL 2025 agent submissions.

## [0.1.1] - 2025-03-16

- Compatibility fixes for newer negmas versions.
- Bugfixes: avoid `log(x)` for `x < 0`; corrected the return type of `get_agents`.
- Reorganized tests out of the main package.
- Code formatting via ruff.

## [0.1.0] - 2024-11-03

- Initial release: ANL 2024 agent submissions and the `anl_agents` registry (`get_agents`).
- Time-limit compatibility updates for submitted agents.
- Bugfixes: infinite recursion in `CARCAgent`; division-by-zero in time-based negotiators.
- Performance improvements via caching for several strategies.
