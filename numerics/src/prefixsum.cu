#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <stdint.h>
#include <iostream>
#include "timer.h"

//[TODO]
//0.performance profile with nsight
//1.optimize shared memory bank conflict
//2.do more work per thread

enum Algo
{
    KoggeStone,
    BrentKung
};

#define THREAD_BLOCK_SIZE 1024

//kogge-Stone scan kernel(one pass)
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

    int i1 = sBid * THREAD_BLOCK_SIZE + threadIdx.x;
    localSum[threadIdx.x] = i1 < len ? array[i1]:0;
    localSum[threadIdx.x + THREAD_BLOCK_SIZE] = 0;

    int pout = 0, pin = 1;
    //Kogge-Stone

#pragma unroll
    for (unsigned int stride = 1; stride < THREAD_BLOCK_SIZE; stride *= 2)
    {
        __syncthreads();

        pout = 1 - pout;
        pin = 1 - pout;

        if (threadIdx.x >= stride)
        {
            localSum[pout*THREAD_BLOCK_SIZE + threadIdx.x] = localSum[pin*THREAD_BLOCK_SIZE + threadIdx.x] + localSum[pin*THREAD_BLOCK_SIZE + threadIdx.x - stride];
        }
        else
        {
            localSum[pout*THREAD_BLOCK_SIZE + threadIdx.x] = localSum[pin*THREAD_BLOCK_SIZE + threadIdx.x];
        }
    }

    __syncthreads();
    if (threadIdx.x == 0)
    {
        if (sBid > 0)
            while (atomicAdd(&blockFlags[sBid], 0) == 0);

        if (sBid > 0)
        {
            scanValues[sBid] = localSum[(pout+1)*THREAD_BLOCK_SIZE - 1] + scanValues[sBid - 1];
        }
        else
        {
            scanValues[sBid] = localSum[(pout+1)*THREAD_BLOCK_SIZE  -1];
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
        if (i1 < len) array[i1] = localSum[threadIdx.x + pout*THREAD_BLOCK_SIZE] + scanValues[sBid - 1];
    }
    else
    {
        if (i1 < len) array[i1] = localSum[threadIdx.x + pout*THREAD_BLOCK_SIZE];
    }
}

//brent-kung
__global__ void bk_scan(int* array, uint64_t len, int* blockFlags, int* scanValues)
{
    const int SECTION_SIZE = THREAD_BLOCK_SIZE * 2;
    __shared__ int sBid;
    __shared__ int localSum[SECTION_SIZE];

    if (threadIdx.x == 0)
    {
        sBid = atomicAdd(&blockFlags[0], 1);
    }
    __syncthreads();

    int idx1 = sBid * SECTION_SIZE + threadIdx.x;
    int idx2 = sBid * SECTION_SIZE + THREAD_BLOCK_SIZE + threadIdx.x;
    localSum[threadIdx.x] = idx1 < len ? array[idx1] : 0;
    localSum[threadIdx.x + THREAD_BLOCK_SIZE] = idx2 < len ? array[idx2] : 0;

    //reduction stage
#pragma unroll
    for (int stride = 1; stride <= THREAD_BLOCK_SIZE; stride *= 2)
    {
        __syncthreads();
        int index = 2 * stride * (threadIdx.x + 1) - 1;
        if (index < SECTION_SIZE)
        {
            localSum[index] += localSum[index - stride];
        }
    }

    //reverse distribution stage
#pragma unroll
    for (int stride = THREAD_BLOCK_SIZE/2; stride > 0; stride /= 2)
    {
        __syncthreads();
        int index = 2 * stride * (threadIdx.x + 1) - 1;
        if (index + stride < SECTION_SIZE)
        {
            localSum[index + stride] += localSum[index];
        }
    }

    __syncthreads();
    if (threadIdx.x == 0)
    {
        if (sBid > 0)
            while (atomicAdd(&blockFlags[sBid], 0) == 0);

        if (sBid > 0)
        {
            scanValues[sBid] = localSum[SECTION_SIZE - 1] + scanValues[sBid - 1];
        }
        else
        {
            scanValues[sBid] = localSum[SECTION_SIZE - 1];
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
        if (idx1 < len) array[idx1] = localSum[threadIdx.x] + scanValues[sBid - 1];
        if (idx2 < len) array[idx2] = localSum[threadIdx.x + THREAD_BLOCK_SIZE] + scanValues[sBid - 1];
    }
    else
    {
        if (idx1 < len) array[idx1] = localSum[threadIdx.x];
        if (idx2 < len) array[idx2] = localSum[threadIdx.x + THREAD_BLOCK_SIZE];
    }
}

void parallel_prefixsum(int* array, uint64_t len, Algo algo)
{
    const int BlockSize = (algo == KoggeStone) ? THREAD_BLOCK_SIZE : (2 * THREAD_BLOCK_SIZE);
    const int NumBlocks = (len + BlockSize - 1) / BlockSize;

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

    if (algo == KoggeStone)
        ks_scan << <NumBlocks, THREAD_BLOCK_SIZE, 2 * sizeof(int) * THREAD_BLOCK_SIZE >> > (d_array, len, d_blockFlags, d_scanValues);
    else
        bk_scan << < NumBlocks, THREAD_BLOCK_SIZE >> > (d_array, len, d_blockFlags, d_scanValues);

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
        //if (block && (i % SECTION_SIZE) == 0) sum = 0;
        sum += array[i];
        array[i] = sum;
    }
}


int main(int argc, char **argv)
{
    Timer timer;

    const int ArraySize = 100000000;
    //const int ArraySize = 16384;
    //const int ArraySize = 2048;
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
    parallel_prefixsum(rawArray1, ArraySize, BrentKung);
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