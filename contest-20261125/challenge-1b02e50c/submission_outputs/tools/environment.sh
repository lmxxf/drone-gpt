# Source from the challenge directory. Read-only reuse of the sibling toolchain.
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PATH="/home/lmxxf/work/drone/contest-20261125/challenge-1b02e50c/tools/bin:$PATH"
export LD_LIBRARY_PATH="/home/lmxxf/work/drone/contest-20261125/challenge-1b02e50c/tools/sysroot/usr/lib/aarch64-linux-gnu:/home/lmxxf/work/drone/contest-20261125/challenge-5f1ea274-3/tools/sysroot/usr/lib/aarch64-linux-gnu/openblas-pthread:/home/lmxxf/work/drone/contest-20261125/challenge-5f1ea274-3/tools/sysroot/usr/lib/aarch64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
