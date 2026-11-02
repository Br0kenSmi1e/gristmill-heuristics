# External-optimizer comparison tooling

Python exporters deriving CC equations with drudge and optimizing them with
gristmill, emitting truthmill candidate `Program` JSONs for the harness'
`--candidate` flag.

Setup (Python 3.12; dummy_spark needs the serial shim):

    uv venv --python 3.12 /tmp/gristmill-venv && . /tmp/gristmill-venv/bin/activate
    uv pip install git+https://github.com/DrudgeCAS/drudge \
                    git+https://github.com/DrudgeCAS/gristmill dummy_spark
    python export_gristmill_ccd.py    # writes benches/candidates/ccd_gristmill.json
    python export_gristmill_ccsd.py   # writes benches/candidates/ccsd_gristmill.json

dspark_shim.py monkeypatches dummy_spark for py3.12 (collections.abc moves,
serial aggregate/broadcast/repartition, etc.).

run_exam.py is the protected milestone-exam driver (PROTOCOL slow loop):
regenerates selected candidates, runs the Axis-0 gate, checks the Axis-1
cap, and times interleaved A/B same-backend runs, writing per-run records
to trials/exam-<git>-runs.jsonl. Run it once per milestone event, on an
idle machine, and commit its outputs with the milestone response.
