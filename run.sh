# alternatively set CUDA_VISIBLE_DEVICES appropriately, see README for details
export CUDA_MANAGED_FORCE_DEVICE_ALLOC=1

# number of CPU threads executing coarse levels
export OMP_NUM_THREADS=4

# enable threads for MVAPICH
# export MV2_ENABLE_AFFINITY=0

# Single GPU
# ./H200build_fp32/bin/hpgmg-fv 4 1

# MPI, one rank per GPU
# mpirun -np 1 ./build/bin/hpgmg-fv 4 1

BINARY=./H200_fp32/bin/hpgmg-fv
RESULTS=results_H200_fp32.txt


echo "==================== 2^2 = 4x4x4 grid ====================" >> $RESULTS
$BINARY 2 1   >> $RESULTS     # 2^2 = 4x4x4 grid
echo "==================== 2^3 = 8x8x8 grid ====================" >> $RESULTS
$BINARY 3 1   >> $RESULTS        # 2^3 = 8x8x8 grid
echo "==================== 2^4 = 16x16x16 grid ====================" >> $RESULTS
$BINARY 4 1   >> $RESULTS        # 2^4 = 16x16x16 grid
echo "==================== 2^5 = 32x32x32 grid ====================" >> $RESULTS
$BINARY 5 1   >> $RESULTS        # 2^5 = 32x32x32 grid
echo "==================== 2^6 = 64x64x64 grid ====================" >> $RESULTS
$BINARY 6 1   >> $RESULTS        # 2^6 = 64x64x64 grid
echo "==================== 2^7 = 128x128x128 grid ====================" >> $RESULTS
$BINARY 7 1   >> $RESULTS        # 2^7 = 128x128x128 grid
echo "==================== 2^8 = 256x256x256 grid ====================" >> $RESULTS
$BINARY 8 1   >> $RESULTS        # 2^8 = 256x256x256 grid
echo "==================== 2^9 = 512x512x512 grid ====================" >> $RESULTS
$BINARY 9 1   >> $RESULTS        # 2^9 = 512x512x512 grid
