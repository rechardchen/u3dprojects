#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <stdint.h>


#define SECTION_SIZE 2048//1024*2


__global__ void stream_scan_kernel(int* array, uint64_t len, int* blockFlags, int* scanValues)
{
    __shared__ int sBid;
    __shared__ int localSum[SECTION_SIZE];

    if (threadIdx.x == 0)
    {
        //blockFlags[0] is the dynamic block index
        sBid = atomicAdd(&blockFlags[0], 1) - 1;
    }
    __syncthreads();

    int i1 = sBid * SECTION_SIZE + threadIdx.x;
    int i2 = i1 + (SECTION_SIZE >> 1);
    localSum[threadIdx.x] = i1 < len ? array[i1]:0;
    localSum[threadIdx.x + (SECTION_SIZE >> 1)] = i2 < len ? array[i2]:0;

    //Kogge-Stone
    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2)
    {
        __syncthreads();

        //Notice: should calculate later slot first
        localSum[threadIdx.x + (SECTION_SIZE >> 1)] += localSum[threadIdx.x + (SECTION_SIZE >> 1) - stride];

        if (threadIdx.x >= stride)
        {
            localSum[threadIdx.x] += localSum[threadIdx.x - stride];
        }
    }

    __syncthreads();
    if (threadIdx.x == 0)
    {
        if (sBid > 0)
            while (atomicAdd(&blockFlags[sBid], 0) == 0);

        if (sBid > 0 && sBid < gridDim.x)
        {
            scanValues[sBid] = localSum[SECTION_SIZE - 1] + scanValues[sBid - 1];
        }
        else if (sBid == 0)
        {
            scanValues[sBid] = localSum[SECTION_SIZE  -1];
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
        if (i1 < len) array[i1] = localSum[threadIdx.x] + scanValues[sBid - 1];
        if (i2 < len) array[i2] = localSum[threadIdx.x + (SECTION_SIZE >> 1)] + scanValues[sBid - 1];
    }
    else
    {
        if (i1 < len) array[i1] = localSum[threadIdx.x];
        if (i2 < len) array[i2] = localSum[threadIdx.x + (SECTION_SIZE >> 1)];
    }
}

void parallel_prefixsum(int* array, uint64_t len)
{
    const int BlockSize = (SECTION_SIZE >> 1);
    const int NumBlocks = (len + SECTION_SIZE - 1) / SECTION_SIZE;

    int* d_array;
    cudaMalloc((void**)&d_array, len * sizeof(int));
    cudaMemcpy(d_array, array, len * sizeof(int), cudaMemcpyHostToDevice);

    int* d_blockFlags;
    cudaMalloc((void**)&d_blockFlags, NumBlocks * sizeof(int));
    cudaMemset(d_blockFlags, 0, NumBlocks * sizeof(int));

    int* d_scanValues;
    cudaMalloc((void**)&d_scanValues, NumBlocks * sizeof(int));
    cudaMemset(d_scanValues, 0, NumBlocks * sizeof(int));

    stream_scan_kernel<<<NumBlocks, BlockSize>>>(d_array, len, d_blockFlags, d_scanValues);

    cudaMemcpy(array, d_array, len * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_array);
    cudaFree(d_blockFlags);
    cudaFree(d_scanValues);
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
    const int ArraySize = 100000000;
    //const int ArraySize = 100;
    int* rawArray = (int*)malloc(ArraySize * sizeof(int));
    for (int i = 0;i < ArraySize;i++)
    {
        rawArray[i] = rand() % 100 - 50;
    }
    int* rawArray1 = (int*)malloc(ArraySize * sizeof(int));
    memcpy(rawArray1, rawArray, ArraySize * sizeof(int));

    prefixsum(rawArray, ArraySize);
    parallel_prefixsum(rawArray1, ArraySize);

    //cudaDeviceSynchronize();
    for (int i = 0;i < ArraySize; ++i)
    {
        rawArray[i] = rawArray[i] - rawArray1[i];
    }

    free(rawArray);
    free(rawArray1);

    return 0;
}