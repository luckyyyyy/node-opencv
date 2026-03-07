# Contributing to node-opencv-rs

Thank you for your interest in contributing! This document explains how to get started, what the conventions are, and what is required before submitting a pull request.

---

## Setting Up the Development Environment

**1. Install system dependencies**

```bash
# Ubuntu / Debian
sudo apt-get install cmake libopencv-dev llvm clang libclang-dev

# macOS
brew install opencv llvm

# Windows — install OpenCV 4, then set:
# OPENCV_INCLUDE_PATHS, OPENCV_LINK_LIBS, OPENCV_LINK_PATHS
```

**2. Install Rust**

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

**3. Clone and build (debug)**

```bash
git clone https://github.com/luckyyyyy/node-opencv.git
cd node-opencv-rs
npm install
npm run build:debug
npm test
```

> On Linux you may need: `export LIBCLANG_PATH=/usr/lib/x86_64-linux-gnu`

---

## Workflow

1. Fork the repository and create a feature branch off `main`.
2. Make your changes.
3. Run `npm run build:debug && npm test` — both must pass before submitting.
4. Open a pull request with a clear description of what was changed and why.

---

## Code Conventions

### Rust

- **No unnecessary `.clone()`** — Never clone data just to satisfy the borrow checker. Restructure ownership or use references instead.
- **No unnecessary locking** — Acquire a lock once, do all work under it, then release. Never lock/unlock the same value multiple times in a single logical operation.
- **Async via `napi::Task` only** — All async operations exposed to JS must use `napi::Task` + `AsyncTask`. Do **not** use `tokio::spawn`, `#[tokio::main]`, or any tokio runtime.
- **`unsafe` blocks must have a `// SAFETY:` comment** explaining the invariant that makes the code safe.
- Run `cargo fmt` and ensure there are no `cargo clippy` warnings before submitting.

### JavaScript / Tests

- Tests live in `tests/*.test.js` and use Node's built-in `node:test` runner.
- Do not use `await` on synchronous functions — it compiles but misleads readers.
- Add test coverage for any new public API.

---

## Branching & Versioning

- Branch names: `feat/<topic>`, `fix/<topic>`, `chore/<topic>`
- **Do not use `git stash`** — commit work-in-progress with a clear message, then amend/squash before the PR is merged.
- Versions follow [Semantic Versioning](https://semver.org). Releases are triggered by pushing a version tag (`v1.2.3`) — the CI/CD pipeline handles building and publishing automatically.

---

## Reporting Issues

Please open an issue if you find a bug or want to request a feature. Include:

- Node.js version (`node --version`)
- Operating system and architecture
- OpenCV version (`pkg-config --modversion opencv4`)
- Minimal reproduction script

---

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](./LICENSE).
