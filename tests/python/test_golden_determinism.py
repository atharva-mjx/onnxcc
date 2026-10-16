#test_golden_determinism.py - checks that the test models and golden outputs are byte identical across two runs
#Registered with ctest in tests/CMakeLists.txt. Exit codes: 0 = pass, 1 = fail, 77 = skipped
#Skips when onnx/numpy/onnxruntime are missing or onnxruntime is not the pinned version,
#unless ONNXCC_REQUIRE_ORT=1 is set (the CI python job sets it so a broken setup fails instead of skipping)

import argparse, filecmp, os, re, shutil, subprocess, sys, tempfile
from pathlib import Path

SKIP=77

def skip(reason):
    if os.environ.get("ONNXCC_REQUIRE_ORT")=="1":
        print(f"FAIL: {reason} (ONNXCC_REQUIRE_ORT=1, so not skipping)")
        sys.exit(1)
    print(f"SKIP: {reason}")
    sys.exit(SKIP)

def fail(reason):
    print(f"FAIL: {reason}")
    sys.exit(1)

def run(cmd, cwd):
    #each run is a separate process, exactly like a user calling the script
    r=subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode!=0:
        fail(f"command failed: {' '.join(map(str, cmd))}\n{r.stdout}{r.stderr}")

def compare_dirs(a, b, what):
    files_a=sorted(p.name for p in a.iterdir())
    files_b=sorted(p.name for p in b.iterdir())
    #guards against two empty folders looking identical
    if not files_a:
        fail(f"{what}: no files were written")
    if files_a!=files_b:
        fail(f"{what}: different file names\n  run 1: {files_a}\n  run 2: {files_b}")
    for name in files_a:
        #shallow=False compares the actual bytes, not just size and timestamps
        if not filecmp.cmp(a/name, b/name, shallow=False):
            fail(f"{what}: {name} differs between runs")
    print(f"ok: {what} ({len(files_a)} files identical)")

def fixture_files(fixtures):
    #only the files generate_test_models.py writes, not golden outputs that may also live in tests/fixtures
    return sorted(list(fixtures.glob("*.onnx"))+list(fixtures.glob("*_input.bin")))

def generate_models(models, repo_root, fixtures, snapshot):
    #delete the old fixtures first so a run that writes nothing cannot pass by reusing old files
    for f in fixture_files(fixtures):
        f.unlink()
    run([sys.executable, models], cwd=repo_root)
    #copy this run's files aside, the next run overwrites tests/fixtures
    snapshot.mkdir(parents=True)
    for f in fixture_files(fixtures):
        shutil.copy(f, snapshot/f.name)

def main():
    repo_default=Path(__file__).resolve().parents[2]
    p=argparse.ArgumentParser(description="Byte-for-byte determinism test for the fixture and golden scripts")
    p.add_argument("--repo-root", default=repo_default, help="onnxcc repository root")
    args=p.parse_args()

    scripts=Path(args.repo_root)/"scripts"
    golden=scripts/"generate_golden.py"
    models=scripts/"generate_test_models.py"
    for f in (golden, models):
        if not f.is_file():
            fail(f"missing {f}")

    #skip cleanly when the python side is not installed
    try:
        import numpy, onnx, onnxruntime
    except ImportError as e:
        skip(f"python dependency not installed ({e.name})")

    #read the pinned version from the golden script so it is defined in one place only
    m=re.search(r'PINNED_ORT\s*=\s*"([^"]+)"', golden.read_text())
    if not m:
        fail("could not find PINNED_ORT in generate_golden.py")
    if onnxruntime.__version__!=m.group(1):
        skip(f"onnxruntime {onnxruntime.__version__} installed, golden script needs {m.group(1)}")

    with tempfile.TemporaryDirectory(prefix="onnxcc_det_") as tmp:
        tmp=Path(tmp)

        #1. test models: generate_test_models.py always writes to tests/fixtures,
        #   so run it twice there and keep a snapshot of each run to compare
        fixtures=Path(args.repo_root)/"tests"/"fixtures"
        snapshots=[]
        for run_id in ("run1", "run2"):
            snapshot=tmp/"models"/run_id
            generate_models(models, args.repo_root, fixtures, snapshot)
            snapshots.append(snapshot)
        compare_dirs(*snapshots, "test models")

        #2. golden outputs for every model, with all intermediates
        onnx_files=sorted(fixtures.glob("*.onnx"))
        if not onnx_files:
            fail(f"generate_test_models.py produced no .onnx files in {fixtures}")
        for model in onnx_files:
            inp=model.with_name(f"{model.stem}_input.bin")
            if not inp.is_file():
                fail(f"no input file {inp.name} for {model.name}")
            out_dirs=[]
            for run_id in ("run1", "run2"):
                out=tmp/"golden"/model.stem/run_id
                run([sys.executable, golden, "--model", model, "--input", inp,
                     "--out-dir", out, "--intermediates"], cwd=tmp)
                out_dirs.append(out)
            compare_dirs(*out_dirs, f"golden outputs for {model.stem}")

    print("PASS: all outputs byte identical across two runs")

if __name__=="__main__":
    main()