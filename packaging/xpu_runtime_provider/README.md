# Comfy Kitchen XPU runtime provider wheel

This builder converts an already-built `comfy-kitchen` XPU wheel into the
co-installable `comfy-kitchen-xpu-runtime` provider distribution. The provider
owns only `comfy_kitchen_xpu_runtime`; its canonical Kitchen implementation is
stored below that package's private `_vendor` directory. It never installs a
top-level `comfy_kitchen` file, so the official distribution can be installed,
reinstalled, or upgraded independently.

Build the normal XPU wheel first, then run:

```bash
python packaging/xpu_runtime_provider/build_wheel.py \
  --source-wheel dist/comfy_kitchen-0.2.33-py3-none-any.whl \
  --output-dir dist/provider \
  --source-revision "$(git rev-parse HEAD)" \
  --torch-version 2.13.0+xpu \
  --xpu-target bmg
```

`--xpu-target` accepts `bmg`, `ptl-h`, or `dg2`. Select the target of the
installed `omni_xpu_kernel` companion wheel; the provider does not compile
native kernels. Target metadata controls activation eligibility, not a promise
that every operator or input is supported. The XPU backend retains its native
capability checks and per-call constraints. An absent optional Sol/CUTE sidecar
does not disable the core operators. DG2 W4A4 matrix multiplication uses the
same-device eager implementation to preserve Kitchen rounding semantics;
preconverted weights require transient format conversion on this route.

The output wheel contains a lightweight entry point in
`comfyui_omnixpu.runtime_providers` plus a manifest recording the canonical
version, exact source revision, source-wheel hash, supported runtime, and every
vendored file hash. Importing the provider metadata does not import PyTorch or
Kitchen. ComfyUI-OmniXPU validates the manifest during prestartup and routes
the canonical `comfy_kitchen` import only when the official version and XPU
runtime match.

Run the portable contract tests in the target development container:

```bash
python -m pytest -q tests/test_xpu_runtime_provider_wheel.py
```
