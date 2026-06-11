# Harbor patch: inject the mutated mini-swe-agent harness (user-approved)

Lets `MiniSweBenchmark` run the optimizer's MUTATED mini-swe-agent codebase
(not upstream git) inside each TB2 task container. Two edits to AHE's pinned
harbor (`.venv/.../harbor/agents/installed/`):

## 1. mini_swe_agent.py — override `MiniSweAgent.setup`
Before the normal install, upload the host harness (advertised by
`MiniSweBenchmark` via the `MSWEA_HARNESS_DIR` env) into the container:

```python
async def setup(self, environment) -> None:
    harness_dir = os.environ.get("MSWEA_HARNESS_DIR", "")
    if harness_dir and Path(harness_dir).is_dir():
        await environment.exec(command="rm -rf /mswea-harness")
        await environment.upload_dir(source_dir=Path(harness_dir), target_dir="/mswea-harness")
    await super().setup(environment)
```

## 2. install-mini-swe-agent.sh.j2 — install from /mswea-harness if present
```bash
if [ -d /mswea-harness ]; then
    uv tool install --reinstall /mswea-harness
else
    uv tool install git+https://github.com/li-boxuan/mini-swe-agent.git   # (or @version)
fi
```

`uv tool install /mswea-harness` installs OUR package, so `mini` uses our code
AND our config (its builtin config dir becomes ours) — no `MSWEA_MINI_CONFIG_PATH`
needed. Curated harness lives at `harness_studio/artifacts/mini_swe_harness/`.
