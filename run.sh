# alternatively set CUDA_VISIBLE_DEVICES appropriately, see README for details
export CUDA_MANAGED_FORCE_DEVICE_ALLOC=1

# number of CPU threads executing coarse levels
export OMP_NUM_THREADS=4

# enable threads for MVAPICH
# export MV2_ENABLE_AFFINITY=0

# Single GPU
# ./H100build/bin/hpgmg-fv 4 1
# ./A100build/bin/hpgmg-fv 4 1

# MPI, one rank per GPU
# mpirun -np 1 ./build/bin/hpgmg-fv 4 1

echo "==================== 2^2 = 4x4x4 grid ====================" >> results_a100_fp64.txt
./A100build_fp64/bin/hpgmg-fv 2 1   >> results_a100_fp64.txt     # 2^2 = 4x4x4 grid
echo "==================== 2^3 = 8x8x8 grid ====================" >> results_a100_fp64.txt
./A100build_fp64/bin/hpgmg-fv 3 1   >> results_a100_fp64.txt        # 2^3 = 8x8x8 grid
echo "==================== 2^4 = 16x16x16 grid ====================" >> results_a100_fp64.txt
./A100build_fp64/bin/hpgmg-fv 4 1   >> results_a100_fp64.txt        # 2^4 = 16x16x16 grid
echo "==================== 2^5 = 32x32x32 grid ====================" >> results_a100_fp64.txt
./A100build_fp64/bin/hpgmg-fv 5 1   >> results_a100_fp64.txt        # 2^5 = 32x32x32 grid   
echo "==================== 2^6 = 64x64x64 grid ====================" >> results_a100_fp64.txt
./A100build_fp64/bin/hpgmg-fv 6 1   >> results_a100_fp64.txt        # 2^6 = 64x64x64 grid
echo "==================== 2^7 = 128x128x128 grid ====================" >> results_a100_fp64.txt
./A100build_fp64/bin/hpgmg-fv 7 1   >> results_a100_fp64.txt        # 2^7 = 128x128x128 grid
echo "==================== 2^8 = 256x256x256 grid ====================" >> results_a100_fp64.txt
./A100build_fp64/bin/hpgmg-fv 8 1   >> results_a100_fp64.txt        # 2^8 = 256x256x256 grid
echo "==================== 2^9 = 512x512x512 grid ====================" >> results_a100_fp64.txt
./A100build_fp64/bin/hpgmg-fv 9 1   >> results_a100_fp64.txt        # 2^9 = 512x512x512 grid 
