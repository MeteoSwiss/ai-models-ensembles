# Source this before any TC tracking work.
#
# Everything lives on CAPSTOR. The previous iopsstor scratch venv
# (venvs/milton_case) and its miniforge3 were destroyed by a scratch purge -
# the purge stripped pyvenv.cfg and every package's top-level files while
# leaving the directory tree, so imports failed with ModuleNotFoundError even
# though site-packages looked populated. Deleted 2026-08-30; do not recreate on
# scratch.
# Overridable off-machine: AIENS_MILTON_CONDA (DetectNodes/StitchNodes +
# udunits) and AIENS_PY (the tracking interpreter).
MILTON_CONDA="${AIENS_MILTON_CONDA:-/capstor/store/cscs/mch/s83/sadamov/miniforge3}"
export LD_LIBRARY_PATH=$MILTON_CONDA/lib:${LD_LIBRARY_PATH:-}
export UDUNITS2_XML_PATH=$MILTON_CONDA/share/udunits/udunits2.xml
export PATH=$MILTON_CONDA/bin:$PATH   # DetectNodes / StitchNodes
export MILTON_PYTHON="${AIENS_PY:-/capstor/store/cscs/mch/s83/sadamov/venvs/ai-models-ensembles/bin/python}"
