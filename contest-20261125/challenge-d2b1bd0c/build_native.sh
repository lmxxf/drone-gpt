#!/usr/bin/env bash
set -euo pipefail
PLASMA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLASMA_BUILD_DIR="$PLASMA_ROOT"
if [[ ! -d "$PLASMA_ROOT/workspace/physics" ]]; then PLASMA_ROOT="$(dirname "$PLASMA_ROOT")"; fi
cd "$PLASMA_ROOT"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p native/debs native/deps native/tmp
export TMPDIR="$PLASMA_ROOT/native/tmp"
python3 -m venv .venv
if [[ -f "$PLASMA_BUILD_DIR/requirements-lock.txt" ]]; then
 .venv/bin/pip install --no-cache-dir -r "$PLASMA_BUILD_DIR/requirements-lock.txt"
elif [[ -f native/requirements-lock.txt ]]; then
 .venv/bin/pip install --no-cache-dir -r native/requirements-lock.txt
else
 .venv/bin/pip install --no-cache-dir numpy scipy shapely netCDF4 h5py cmake
fi
cd native/debs
taskset -c 4-7 apt-get download libhdf5-dev libhdf5-103-1t64 libhdf5-fortran-102t64 libhdf5-hl-100t64 libhdf5-hl-fortran-100t64 libmetis-dev libmetis5 libsuperlu-dev libsuperlu6 libarpack2-dev libarpack2t64 libnetcdf-dev libnetcdf19t64 libnetcdff-dev libnetcdff7 libblas-dev liblapack-dev libaec0 libsz2
for package in *.deb; do
    # Experimental OpenBLAS is deliberately excluded: use reference BLAS.
    [[ "$package" == libopenblas* ]] && continue
    dpkg-deb -x "$package" ../deps
done
cd "$PLASMA_ROOT"
rm -f native/deps/usr/lib/aarch64-linux-gnu/libblas.so.3 native/deps/usr/lib/aarch64-linux-gnu/liblapack.so.3
ln -sf /usr/lib/aarch64-linux-gnu/libblas.so.3 native/deps/usr/lib/aarch64-linux-gnu/libblas.so
ln -sf /usr/lib/aarch64-linux-gnu/liblapack.so.3 native/deps/usr/lib/aarch64-linux-gnu/liblapack.so
export LD_LIBRARY_PATH="$PLASMA_ROOT/native/deps/usr/lib/aarch64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [[ ! -d native/OpenFUSIONToolkit ]]; then tar xzf datasets/*/1/resources/upstream/oft-source.tar.gz -C native; fi
if [[ ! -d native/GPEC ]]; then tar xzf datasets/*/1/resources/upstream/gpec-source.tar.gz -C native; fi
# Noble's HDF5 1.10 Fortran module lacks the optional persistent-space strategy constant.
sed -i 's/if( HDF5_VERSION VERSION_GREATER_EQUAL "1.10.0" )/if( HDF5_VERSION VERSION_GREATER_EQUAL "1.12.0" )/' native/OpenFUSIONToolkit/src/CMakeLists.txt
taskset -c 4-7 .venv/bin/cmake -S native/OpenFUSIONToolkit/src -B native/oft-build -G Ninja \
 -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PLASMA_ROOT/native/oft-install" \
 -DCMAKE_PREFIX_PATH="$PLASMA_ROOT/native/deps/usr" \
 -DCMAKE_LIBRARY_PATH="$PLASMA_ROOT/native/deps/usr/lib/aarch64-linux-gnu" \
 -DBLAS_LIBRARIES=/usr/lib/aarch64-linux-gnu/libblas.so.3 \
 -DLAPACK_LIBRARIES=/usr/lib/aarch64-linux-gnu/liblapack.so.3 \
 -DHDF5_C_LIBRARY_hdf5="$PLASMA_ROOT/native/deps/usr/lib/aarch64-linux-gnu/libhdf5_serial.so" \
 -DHDF5_Fortran_LIBRARY_hdf5_fortran="$PLASMA_ROOT/native/deps/usr/lib/aarch64-linux-gnu/libhdf5_serial_fortran.so" \
 -DHDF5_INCLUDE_DIRS="$PLASMA_ROOT/native/deps/usr/include/hdf5/serial" \
 -DOFT_SUPERLU_ROOT="$PLASMA_ROOT/native/deps/usr" \
 -DSUPERLU_INCLUDE_DIR="$PLASMA_ROOT/native/deps/usr/include/superlu" \
 -DOFT_ARPACK_ROOT="$PLASMA_ROOT/native/deps/usr" -DOFT_USE_LTO=OFF -DOFT_PACKAGE_PYTHON=OFF
for target in oftbase oftgrid oftla oftfem oftphysics all; do
 taskset -c 4-7 .venv/bin/cmake --build native/oft-build --target "$target" -j4
done
printf '%s\n' "$PLASMA_ROOT/native/oft-build/python" > .venv/lib/python3.12/site-packages/plasma_oft.pth
taskset -c 4-7 make -C native/GPEC/install -j4 dcon FC=gfortran CC=gcc \
 LAPACKHOME="$PLASMA_ROOT/native/deps/usr/lib/aarch64-linux-gnu" \
 NETCDFHOME="$PLASMA_ROOT/native/deps/usr/lib/aarch64-linux-gnu" \
 NETCDFINC="$PLASMA_ROOT/native/deps/usr/include"
