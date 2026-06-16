# Contributing

Thanks for your interest in contributing to GPUArrays.jl.

GPUArrays.jl provides shared array functionality for Julia GPU backends. The most useful contributions are small, focused changes that keep backend compatibility in mind and include enough context for maintainers to understand which array behavior or backend interaction is affected.

## Getting Started

1. Install Julia.
2. Fork and clone the repository.
3. Create a branch for your change.

```bash
git clone https://github.com/JuliaGPU/GPUArrays.jl.git
cd GPUArrays.jl
git checkout -b your-change
```

## Local Checks

The GitHub Actions test workflow develops the local subpackages before running the test suite. To mirror that setup locally, run:

```bash
julia --project=. -e 'using Pkg; Pkg.develop([PackageSpec(path="lib/GPUArraysCore"), PackageSpec(path="lib/JLArrays")]); Pkg.test(; test_args=["--verbose"])'
```

To build the documentation locally, instantiate the docs environment and run the docs build:

```bash
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
julia --project=docs docs/make.jl
```

The repository also has backend-specific CI coverage for packages such as CUDA.jl, AMDGPU.jl, oneAPI.jl, Metal.jl, and OpenCL.jl. If your change depends on a specific GPU backend, mention that in the pull request.

## Pull Requests

Please keep pull requests focused and include:

- what behavior changed
- why the change is needed
- which local checks you ran
- whether the change is backend-specific or affects shared GPU array behavior

For larger changes, open an issue first so maintainers can discuss the design before review time is spent on implementation details.
