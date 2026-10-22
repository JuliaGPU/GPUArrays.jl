# oneAPI sparse support artifact

The sparse migration requires native wrapper contract version2, not merely the
presence of wide CSC symbols. Published oneAPI_Support0.11.0+0 has those symbols
but lacks the catch-all exception boundary. oneAPI rejects it for sparse execution
without disabling dense operations. The runtime marker guard remains necessary
for preferences and custom overrides after a new minimum version is registered.

## Recipe and generated artifact

Recipe: maleadt/Yggdrasil branch tb/oneapi-sparse-v13-support, commit
31c79455bc9f0b415a75ceb9faac59a26148e8f4, based on master0defdb4d8. Independent
e0dc7751df3b832bdc67f0061b01c0a3a727ee2e fixes an existing parse error. The recipe
keeps upstream oneAPI source5093529311e36f530c5990e4b0b3669452c45d8c and Intel2026.1,
applying only the native cpp/header patch. Patched files compare equal to tested
oneAPId1cb45d (runtime-equivalent successor c7e27cb). Dependency floors are the
validated Loader1.32/OpenCL2024.10.24/Hwloc2.15, not asserted minimum ABI versions.

Pegasus crossbuild, Julia1.13.2 with the Yggdrasil root project:

- x86_64-linux-gnu-cxx11, version0.11.1+0; compiler/link/install completed, exit0.
- Artifact tree876c51345ae21089df99bd35dfc29cf292c2a4ce.
- Tarball SHA25622eadbfd15512fa8aa9867970f96c1907ce5314e767144a33f44410a1b8963eb.
- Loaded support SHA256d5e977efdfe577477c74827cd9b827b61d14f49fa2c8466fb3387a3e3c933fd7.
- Recipe products live under ~/Julia/tmp/yggdrasil-oneapi-v13/O/oneAPI_Support/products.
- Generated wrapper is ~/Julia/tmp/ga13-ygg-jll-dev/oneAPI_Support_jll on both hosts;
  local file downloads are not a published/registered artifact.
- Full build log: ~/Julia/tmp/ga13-ygg-build-no-update.log. Metadata-only generation
  passed separately, but is not the crossbuild result.

Run the recipe with --verbose --deploy=local x86_64-linux-gnu-cxx11 through a unique
jld daemon, --no-revise --idle-timeout=2h. Clear ARGS first. Set
BinaryBuilder.devdir[] to a task-owned directory after importing BinaryBuilder:
the precompiled Ref can retain ~/.julia/dev despite JULIA_PKG_DEVDIR. The newly
generated wrapper was moved recoverably into task scratch; existing pkg checkouts
were not modified. No artifact was uploaded or registered.

## Runtime validation without a source override

Taurus Arc A770, GPUArrays4e20de2 and oneAPId1cb45d, exact AK/Atomix migration pins.
Fresh environments explicitly develop the generated JLL and all coordinated packages;
there is no liboneapi_support_path preference. Actual loaded path is
~/.julia/artifacts/876c51345ae21089df99bd35dfc29cf292c2a4ce/lib/liboneapi_support.so.
Artifact identity, marker eligibility and device functionality pass three checks.

Julia1.10.12 actual Pkg.test, allow_reresolve=false: native oneMKL1279,
resources121, ABI96 pass, plus24 pre-existing broken CSC checks. Total1496pass/24broken;
no shared-suite pass is claimed for this run. Log:
~/Julia/tmp/ga13-oneapi-generated-jll110-env/test.log.

Julia1.13.2 actual Pkg.test passes13167, plus24 pre-existing broken: shared sparse11671,
native oneMKL1279, resources121, ABI96. Log:
~/Julia/tmp/ga13-oneapi-generated-jll-env/test.log. Do not confuse the earlier separately
source-built support result13138pass/24broken with this generated artifact. The final
all-dimensional broadcast guard postdates this frozen GA source: its2058 local
regressions,247contracts and focused groups pass separately at34d125e. It does not
change the native support ABI; final physical guard checks remain separate evidence.

Final physical guard checks at34d125e/oneAPI06f2f77 pass244+ambiguity1 using the same
generated artifact. Separately, the actual final pinned helper9a9126f/06f2f77 builds
support from source and passes13231+24broken: shared11734,native1279,resources122,
ABI96. Loaded support SHA256b3f319f2972b03890f14b97c0616c67992ccaa1662075136db55fd684c89cd9a;
log ~/Julia/tmp/ga13-oneapi-generated-jll-env/final9a-root.log. This verifies actual
checked setup/Pkg.test plumbing; it remains distinct from the generated JLL and CI.

## Inherited audit warnings and release gates

BinaryBuilder's binutils2.25.1 objdump aborts while auditing some Intel libraries.
Standalone vendor MKL dlopen probes also report undefined symbols, and the linker
script libmkl_sycl.so is not itself an ELF library. The baseline0.11.0 recipe at
e0dc775 (only the parse fix, no native patch/version/compat changes) reproduces these
warnings and produces its tarball. Log: ~/Julia/tmp/ga13-ygg-baseline-build.log.

This establishes that the warnings are inherited, not that the interrupted audits
passed. No audit bypass or soft installation failure was added. Normal generated-JLL
loading/native operations are tested separately above. Resolve or explicitly review
the incomplete audit checks before artifact publication, then test the published
artifact and raise oneAPI's compatibility minimum to the actual registered0.11.1.
Until then the source-built CI override and exact stack pins are temporary validation
plumbing, not normal registered release resolution.
