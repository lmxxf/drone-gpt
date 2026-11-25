# Source from the challenge root. Libraries and executables remain local.
CHALLENGE_ROOT=/home/lmxxf/work/drone/contest-20261125/challenge-5f1ea274-3
export PATH="$CHALLENGE_ROOT/tools/bin:$CHALLENGE_ROOT/.venv/bin:$PATH"
export LD_LIBRARY_PATH="$CHALLENGE_ROOT/tools/sysroot/usr/lib/aarch64-linux-gnu:$CHALLENGE_ROOT/tools/sysroot/usr/lib/aarch64-linux-gnu/openblas-pthread${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export UV_CACHE_DIR="$CHALLENGE_ROOT/tools/cache/uv"
export MPLCONFIGDIR="$CHALLENGE_ROOT/tools/cache/matplotlib"
export XDG_CACHE_HOME="$CHALLENGE_ROOT/tools/cache"

export OMPI_MCA_mpi_yield_when_idle=1 OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=0
