#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <stdint.h>
#include <iostream>
#include "timer.h"

#define SECTION_SIZE 1024

//kogge-Stone scan kernel
__global__ void ks_scan(int* array, uint64_t len, int* blockFlags, int* scanValues)
{
    __shared__ int sBid;
    extern __shared__ int localSum[];

    if (threadIdx.x == 0)
    {
        //blockFlags[0] is the dynamic block index
        sBid = atomicAdd(&blockFlags[0], 1);
    }
    __syncthreads();

    int i1 = sBid * SECTION_SIZE + threadIdx.x;
    localSum[threadIdx.x] = i1 < len ? array[i1]:0;
    localSum[threadIdx.x + SECTION_SIZE] = 0;

    int pout = 0, pin = 1;
    //Kogge-Stone
    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2)
    {
        __syncthreads();

        pout = 1 - pout;
        pin = 1 - pout;

        if (threadIdx.x >= stride)
        {
            localSum[pout*SECTION_SIZE + threadIdx.x] = localSum[pin*SECTION_SIZE + threadIdx.x] + localSum[pin*SECTION_SIZE + threadIdx.x - stride];
        }
        else
        {
            localSum[pout*SECTION_SIZE + threadIdx.x] = localSum[pin*SECTION_SIZE + threadIdx.x];
        }
    }

    __syncthreads();
    if (threadIdx.x == 0)
    {
        if (sBid > 0)
            while (atomicAdd(&blockFlags[sBid], 0) == 0);

        if (sBid > 0)
        {
            scanValues[sBid] = localSum[(pout+1)*SECTION_SIZE - 1] + scanValues[sBid - 1];
        }
        else
        {
            scanValues[sBid] = localSum[(pout+1)*SECTION_SIZE  -1];
        }

        __threadfence();
        if (sBid < gridDim.x - 1)
        {
            atomicAdd(&blockFlags[sBid + 1], 1);
        }
    }

    __syncthreads();
    if (sBid > 0)
    {
        if (i1 < len) array[i1] = localSum[threadIdx.x + pout*SECTION_SIZE] + scanValues[sBid - 1];
    }
    else
    {
        if (i1 < len) array[i1] = localSum[threadIdx.x + pout*SECTION_SIZE];
    }
}

void parallel_prefixsum(int* array, uint64_t len)
{
    const int BlockSize = SECTION_SIZE;
    const int NumBlocks = (len + SECTION_SIZE - 1) / SECTION_SIZE;

    int* d_array;
    cudaMalloc((void**)&d_array, len * sizeof(int));
    cudaMemcpy(d_array, array, len * sizeof(int), cudaMemcpyHostToDevice);
    //cudaMallocAsync((void**)&d_array, len * sizeof(int), 0);
    //cudaMemcpyAsync(d_array, array, len * sizeof(int), cudaMemcpyHostToDevice);

    int* d_blockFlags;
    cudaMalloc((void**)&d_blockFlags, NumBlocks * sizeof(int));
    cudaMemset(d_blockFlags, 0, NumBlocks * sizeof(int));
    //cudaMallocAsync((void**)&d_blockFlags, NumBlocks * sizeof(int), 0);
    //cudaMemsetAsync(d_blockFlags, 0, NumBlocks * sizeof(int));

    int* d_scanValues;
    cudaMalloc((void**)&d_scanValues, NumBlocks * sizeof(int));
    cudaMemset(d_scanValues, 0, NumBlocks * sizeof(int));
    //cudaMallocAsync((void**)&d_scanValues, NumBlocks * sizeof(int), 0);
    //cudaMemsetAsync(d_scanValues, 0, NumBlocks * sizeof(int));

    //k-s scan, double buffer shared memory
    ks_scan << <NumBlocks, BlockSize, 2 * sizeof(int) * SECTION_SIZE >> > (d_array, len, d_blockFlags, d_scanValues);

    cudaMemcpy(array, d_array, len * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_array);
    cudaFree(d_blockFlags);
    cudaFree(d_scanValues);
    //cudaFreeAsync(d_array, 0);
    //cudaFreeAsync(d_blockFlags,0);
    //cudaFreeAsync(d_scanValues,0);
}

void prefixsum(int* array, uint64_t len)
{
    int sum = 0;
    for (uint64_t i = 0; i < len; i++)
    {
        sum += array[i];
        array[i] = sum;
    }
}


int main(int argc, char **argv)
{
    Timer timer;

    const int ArraySize = 100000000;
    //const int ArraySize = 16384;
    //const int ArraySize = 100;
    int* rawArray = (int*)malloc(ArraySize * sizeof(int));
    for (int i = 0;i < ArraySize;i++)
    {
        rawArray[i] = rand() % 100 - 50;
    }
    int* rawArray1 = (int*)malloc(ArraySize * sizeof(int));
    memcpy(rawArray1, rawArray, ArraySize * sizeof(int));

    timer.start();
    prefixsum(rawArray, ArraySize);
    timer.stop();
    std::cout << "CPU prefixsum time: " << timer.microseconds() << std::endl;

    timer.start();
    parallel_prefixsum(rawArray1, ArraySize);
    timer.stop();
    std::cout << "GPU prefixsum time: " << timer.microseconds() << std::endl;

    cudaDeviceSynchronize();
    bool correct = true;
    for (int i = 0;i < ArraySize; ++i)
    {
        if (rawArray[i] != rawArray1[i])
        {
            std::cerr << "rawArray[" << i << "] = " << rawArray[i] << ", rawArray1[" << i << "] = " << rawArray1[i] << std::endl;
            correct = false;
            break;
        }
    }
    std::cout << "result is " << (correct ? "correct" : "wrong") << std::endl;

    free(rawArray);
    free(rawArray1);

    return 0;
}