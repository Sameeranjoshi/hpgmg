#!/bin/bash
# run_experiments.sh — Run all 4 configs (pre/post x bottom) and produce summary table
# Configs: 4_4_6, 4_4_100, 6_6_6, 6_6_100
# Usage: ./run_experiments.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ---- Source files that need patching ----
FV2_CPU="finite-volume/source/operators.fv2.c"
FV2_GPU="finite-volume/source/cuda/operators.fv2.cu"
SOLVERS="finite-volume/source/solvers.c"

# ---- GPU environment ----
export CUDA_MANAGED_FORCE_DEVICE_ALLOC=1
export OMP_NUM_THREADS=4

# ---- Build settings (same as build.sh) ----
CC=$(which mpicc)
NVCC=$(which nvcc)
CUDA_ARCH="-gencode arch=compute_90,code=sm_90"
LDLIBS="-ldl"

OPTS=""
OPTS+="-DBLOCKCOPY_TILE_I=32 "
OPTS+="-DBLOCKCOPY_TILE_J=4 "
OPTS+="-DBLOCKCOPY_TILE_K=8 "
OPTS+="-DBOUNDARY_TILE_I=64 "
OPTS+="-DBOUNDARY_TILE_J=16 "
OPTS+="-DBOUNDARY_TILE_K=16 "
OPTS+="-DHOST_LEVEL_SIZE_THRESHOLD=0 "
OPTS+="-DMAX_SOLVES=10 "
OPTS+="-DCUDA_UM_ALLOC "
OPTS+="-DCUDA_UM_ZERO_COPY "
OPTS+="-DMPI_ALLOC_ZERO_COPY "
OPTS+="-DUSE_REG "
OPTS+="-DUSE_TEX "
OPTS+="-DMPICH_IGNORE_CXX_SEEK "
OPTS+="-DMPICH_SKIP_MPICXX "
OPTS+="-DUSE_DIRICHLET_BC "

# ---- Configs: pre_post_bottom ----
CONFIGS="4_4_6 4_4_100 6_6_6 6_6_100"
RESULTS_DIR="$SCRIPT_DIR/paper_results_sc"
mkdir -p "$RESULTS_DIR"

# ---- Helper: patch NUM_SMOOTHS in a file for USE_JACOBI ----
# Matches the line: #define NUM_SMOOTHS      <N>
# that appears right after #elif   USE_JACOBI (CPU) or #elif   USE_JACOBI (GPU)
patch_num_smooths() {
    local file="$1"
    local value="$2"
    # For CPU file (operators.fv2.c): two lines, USE_JACOBI and USE_L1JACOBI
    # For GPU file (cuda/operators.fv2.cu): two lines, USE_JACOBI and USE_L1JACOBI
    # Replace all NUM_SMOOTHS lines that have value 4 or 6 under USE_JACOBI/USE_L1JACOBI
    # Strategy: use sed to replace the NUM_SMOOTHS after USE_JACOBI and USE_L1JACOBI
    sed -i "/USE_JACOBI/,/NUM_SMOOTHS/{s/#define NUM_SMOOTHS.*[0-9]/#define NUM_SMOOTHS      $value/}" "$file"
    sed -i "/USE_L1JACOBI/,/NUM_SMOOTHS/{s/#define NUM_SMOOTHS.*[0-9]/#define NUM_SMOOTHS      $value/}" "$file"
}

# ---- Helper: patch maxSmoothsBottom in solvers.c ----
patch_bottom_iters() {
    local value="$1"
    sed -i "s/maxSmoothsBottom=[0-9]*/maxSmoothsBottom=$value/" "$SOLVERS"
}

# ---- Build and run one config ----
run_config() {
    local pre_post="$1"
    local bottom="$2"
    local config_name="${pre_post}_${pre_post}_${bottom}"
    local build_dir="build_${config_name}"
    local results_file="${RESULTS_DIR}/glow_${config_name}.txt"

    echo "============================================================"
    echo "  Config: pre/post=${pre_post}, bottom=${bottom}"
    echo "  Build dir: ${build_dir}"
    echo "  Results: ${results_file}"
    echo "============================================================"

    # Step 1: Patch source code
    echo "[1/4] Patching NUM_SMOOTHS=${pre_post}, maxSmoothsBottom=${bottom}..."
    patch_num_smooths "$FV2_CPU" "$pre_post"
    patch_num_smooths "$FV2_GPU" "$pre_post"
    patch_bottom_iters "$bottom"

    # Step 2: Build
    echo "[2/4] Building in ${build_dir}..."
    rm -rf "$build_dir"
    python ./configure --arch="$build_dir" \
        --CC=$CC --NVCC=$NVCC \
        --CFLAGS="-O2 -fopenmp $OPTS" \
        --NVCCFLAGS="-O2 -lineinfo $OPTS" \
        --CUDAARCH="$CUDA_ARCH" \
        --LDLIBS="$LDLIBS" \
        --no-fe --no-fv-mpi --fv-cycle="V" --fv-smoother="jacobi"
    make -j$(nproc) -C "$build_dir"

    # Step 3: Run all problem sizes
    local binary="./${build_dir}/bin/hpgmg-fv"
    echo "[3/4] Running experiments..."
    > "$results_file"  # clear results file

    for log2 in 2 3 4 5 6 7 8 9; do
        local gridsize=$((1 << log2))
        echo "  Running ${gridsize}^3 grid..."
        echo "==================== 2^${log2} = ${gridsize}x${gridsize}x${gridsize} grid ====================" >> "$results_file"
        $binary $log2 1 >> "$results_file" 2>&1
    done

    echo "[4/4] Done with config ${config_name}."
    echo ""
}

# ---- Main: run all configs ----
echo "Starting HPGMG experiments: $CONFIGS"
echo "Results directory: $RESULTS_DIR"
echo ""

for config in $CONFIGS; do
    pre_post="${config%%_*}"         # e.g. "4" from "4_4_6"
    bottom="${config##*_}"           # e.g. "6" from "4_4_6"
    run_config "$pre_post" "$bottom"
done

# ---- Extract results and build summary table ----
echo ""
echo "============================================================"
echo "  Extracting results and building summary table"
echo "============================================================"
echo ""

# Python script for parsing and table generation
python3 - "$RESULTS_DIR" "$CONFIGS" <<'PYEOF'
import sys, re, os

results_dir = sys.argv[1]
configs = sys.argv[2].split()

log2_sizes = [4, 5, 6, 7, 8, 9]

def parse_results(filepath):
    """For each grid size, get MGSolve time and vcycles from the FIRST 'Running' block only."""
    data = {}
    with open(filepath) as f:
        text = f.read()

    # Split by grid headers: ==================== 2^N = NxNxN grid ====================
    grid_sections = re.split(r'={3,}\s*2\^(\d+)\s*=\s*\d+x\d+x\d+\s*grid\s*={3,}', text)
    # Result: [preamble, log2_1, section_1, log2_2, section_2, ...]

    for i in range(1, len(grid_sections), 2):
        log2 = int(grid_sections[i])
        section = grid_sections[i + 1]

        # Split by "Running N solves" headers to get sub-runs
        running_blocks = re.split(r'={3,}\s*Running\s+\d+\s+solves\s*={3,}', section)
        # running_blocks[0] = warmup, [1] = first timed sub-run, [2] = second, etc.

        if len(running_blocks) < 2:
            continue

        first_run = running_blocks[1]  # FIRST timed sub-run only

        mg_match = re.search(r'Total time in MGSolve\s+([\d.]+)', first_run)
        vc_match = re.search(r'number of v-cycles\s+(\d+)', first_run)
        bi_match = re.search(r'Bottom solver iterations\s+(\d+)', first_run)

        if mg_match and vc_match:
            mgsolve = float(mg_match.group(1))
            vcycles = int(vc_match.group(1))
            bottom = int(bi_match.group(1)) if bi_match else 0
            data[log2] = {
                'mgsolve': mgsolve,
                'vcycles': vcycles,
                'bottom_iters': bottom,
                'time_per_vcycle': mgsolve / vcycles if vcycles > 0 else 0,
            }
    return data

# Parse all configs, preserve order: 6_6_100, 4_4_6, 6_6_6, 4_4_100
config_order = ["6_6_100", "4_4_6", "6_6_6", "4_4_100"]
display_names = {"6_6_100": "H200(6/6/100)", "4_4_6": "H200(4/4/6)",
                 "6_6_6": "H200(6/6/6)", "4_4_100": "H200(4/4/100)"}

all_data = {}
for config in config_order:
    pp = config.split('_')[0]
    bot = config.split('_')[-1]
    cn = f"{pp}_{pp}_{bot}"
    fp = os.path.join(results_dir, f"glow_{cn}.txt")
    if os.path.exists(fp):
        all_data[cn] = parse_results(fp)
    else:
        print(f"WARNING: {fp} not found, skipping.")

if not all_data:
    print("No results found!")
    sys.exit(1)

ordered_keys = [c.split('_')[0]+"_"+c.split('_')[0]+"_"+c.split('_')[-1] for c in config_order]
ordered_keys = [k for k in ordered_keys if k in all_data]

# ---- Time per V-cycle table ----
print("=" * 100)
print("Time per V-cycle = MGSolve / vcycles (first sub-run per grid)")
print("=" * 100)
header = f"{'Grid':>12}"
for cn in ordered_keys:
    header += f" | {display_names[cn]:>16}"
print(header)
print("-" * len(header))
for log2 in log2_sizes:
    gs = 1 << log2
    row = f"{gs:>5}^3 ({log2})"
    for cn in ordered_keys:
        d = all_data[cn].get(log2)
        if d:
            row += f" | {d['time_per_vcycle']:>16.6f}"
        else:
            row += f" | {'N/A':>16}"
    print(row)
print()

# ---- Detail table ----
print("=" * 100)
print("Detail: MGSolve(s) / vcycles / bottom_iters (first sub-run per grid)")
print("=" * 100)
header = f"{'Grid':>12}"
for cn in ordered_keys:
    header += f" | {'MGSolve':>10} {'Vc':>3} {'Bot':>5}"
print(header)
print("-" * len(header))
for log2 in log2_sizes:
    gs = 1 << log2
    row = f"{gs:>5}^3 ({log2})"
    for cn in ordered_keys:
        d = all_data[cn].get(log2)
        if d:
            row += f" | {d['mgsolve']:>10.6f} {d['vcycles']:>3d} {d['bottom_iters']:>5d}"
        else:
            row += f" | {'N/A':>10} {'':>3} {'':>5}"
    print(row)
print()

PYEOF

echo ""
echo "All done. Raw results are in: $RESULTS_DIR/"
