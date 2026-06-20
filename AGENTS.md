# AGENTS.md

Guidance for AI agents working in this repository.

## API Design Principles

**Prioritize long-term API quality over short-term convenience.**

This project is in early development. Breaking changes are acceptable — refactor callers when needed rather than compromising the API to preserve backwards compatibility. Named return objects (e.g. `OptimizationResult`) are preferred over positional tuples even when they require migrating many call sites.
