/*
 * Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

/* This sample queries the properties of the CUDA devices present in the system. */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>


inline int _ConvertSMVer2Cores(int major, int minor)
{
    // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
    typedef struct
    {
        int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] =
    {
        { 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
        { 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
        { 0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
        { 0x32, 192}, // Kepler Generation (SM 3.2) GK10x class
        { 0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
        { 0x37, 192}, // Kepler Generation (SM 3.7) GK21x class
        { 0x50, 128}, // Maxwell Generation (SM 5.0) GM10x class
        { 0x52, 128}, // Maxwell Generation (SM 5.2) GM20x class
        { 0x53, 128}, // Maxwell Generation (SM 5.3) GM20x class
        { 0x60, 64 }, // Pascal Generation (SM 6.0) GP100 class
        { 0x61, 128}, // Pascal Generation (SM 6.1) GP10x class
        { 0x62, 128}, // Pascal Generation (SM 6.2) GP10x class
        {   -1, -1 }
    };

    int index = 0;

    while (nGpuArchCoresPerSM[index].SM != -1)
    {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
        {
            return nGpuArchCoresPerSM[index].Cores;
        }

        index++;
    }

    // If we don't find the values, we default use the previous one to run properly
    printf("MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n", major, minor, nGpuArchCoresPerSM[index-1].Cores);
    return nGpuArchCoresPerSM[index-1].Cores;
}







////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv)
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0)
        printf("There is no device supporting CUDA\n");
    int dev;
    for (dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        if (dev == 0) {
            if (deviceProp.major == 9999 && deviceProp.minor == 9999)
                printf("There is no device supporting CUDA.\n");
            else if (deviceCount == 1)
                printf("There is 1 device supporting CUDA\n");
            else
                printf("There are %d devices supporting CUDA\n", deviceCount);
        }
        printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);
        printf("  Major revision number:                         %d\n",
               deviceProp.major);
        printf("  Minor revision number:                         %d\n",
               deviceProp.minor);
        printf("  Total amount of global memory:                 %zd bytes\n",
               deviceProp.totalGlobalMem);
    #if CUDART_VERSION >= 2000
        printf("  Number of multiprocessors:                     %d\n",
               deviceProp.multiProcessorCount);
        printf("  CUDA Cores/MP:                                 %d\n",
               _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor));
        printf("  Total CUDA Cores                               %d\n",
               _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount);
    #endif
        printf("  Total amount of constant memory:               %zd bytes\n",
               deviceProp.totalConstMem);
        printf("  Total amount of shared memory per block:       %zd bytes\n",
               deviceProp.sharedMemPerBlock);
        printf("  Total number of registers available per block: %d\n",
               deviceProp.regsPerBlock);
        printf("  Warp size:                                     %d\n",
               deviceProp.warpSize);
        printf("  Maximum number of threads per block:           %d\n",
               deviceProp.maxThreadsPerBlock);
        printf("  Maximum sizes of each dimension of a block:    %d x %d x %d\n",
               deviceProp.maxThreadsDim[0],
               deviceProp.maxThreadsDim[1],
               deviceProp.maxThreadsDim[2]);
        printf("  Maximum sizes of each dimension of a grid:     %d x %d x %d\n",
               deviceProp.maxGridSize[0],
               deviceProp.maxGridSize[1],
               deviceProp.maxGridSize[2]);
        printf("  Maximum memory pitch:                          %zd bytes\n",
               deviceProp.memPitch);
        printf("  Texture alignment:                             %zd bytes\n",
               deviceProp.textureAlignment);
        printf("  Clock rate:                                    %.2f GHz\n",
               deviceProp.clockRate * 1e-6f);
        printf("  Memory Clock rate:                             %.2f GHz\n",
               deviceProp.memoryClockRate * 1e-6f);
        printf("  Memory Bus Width:                              %d bits\n",
               deviceProp.memoryBusWidth);
        printf("  Number of asynchronous engines:                %d\n",
               deviceProp.asyncEngineCount);
        printf("  It can execute multiple kernels concurrently:  %s\n",
               deviceProp.concurrentKernels ? "Yes" : "No");
    #if CUDART_VERSION >= 2000
        printf("  Concurrent copy and execution:                 %s\n",
               deviceProp.deviceOverlap ? "Yes" : "No");
    #endif
    }
    printf("\nTEST PASSED\n");
}
