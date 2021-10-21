#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <vector>
#include <string>
#include <fstream>
#include <assert.h>
#include <algorithm>

#include "device_launch_parameters.h"
#include "./src/utils.h"

constexpr int GRID_DEFAULT_SIZE = 1024;
constexpr int BLOCK_DEFAULT_SIZE = 512;
constexpr int FOLD = 32;
extern const int warpSize; //32
#define FULL_MASK 0xffffffff

enum cluster_kernel_t {cpu,naive,SharedMemory,SharedMemory2, ParallelReduction,MoreParallelReduction};
enum dist_kernel_t {dst1,dst2,dst3,dst31,dst32,dst4};

template<typename ValueType, typename IndexType>
int read_csr(std::string path, ValueType *cv, IndexType *rp, IndexType *ci){
    std::ifstream f1((path+"tcsr_values.txt"), std::ifstream::in);
    std::ifstream f2((path+"tcsr_col.txt"), std::ifstream::in);
    std::ifstream f3((path+"tcsr_ptr.txt"), std::ifstream::in);
    int count;
    ValueType value;
    IndexType idx;

    if(f1){
        for(count = 0;f1>>value; count++){
            cv[count] = value;
        }
        f1.close(); //pin(cv,1,num_nz,__LINE__);
        //    num_nz = count;
    }
    if(f2){
        for(count = 0; f2>>idx; count++){
            ci[count] = idx;
        }
        f2.close();//pin(ci,1,num_nz,__LINE__);
    }
    else{
        std::cout<<"fail to open col_idx!"<<std::endl;
    }

    if(f3){
        for(count = 0; f3>>idx ;count++){
            rp[count] = idx;
        }
        f3.close();
    }
    else{
        std::cout<<"fail to open row_ptrs!"<<std::endl;
    }
    return 0;
}


int read_Membership(std::string path, int *Membership){
    std::ifstream f1(path,std::ifstream::in);
    int tmp = 0;
    while(f1>>tmp){
        *(Membership++)=tmp;
    }
    f1.close();
    return 0;
}

template<typename ValueType>
int get_nz(std::string path, int &nz){
    std::ifstream ff;
    ff.open((path+"tcsr_ptr.txt"), std::ifstream::in);
    int count = 0;
    ValueType tmp;
    while(ff>>tmp){
        count++;
    }
    ff.close();
    nz = tmp;
    return 0;
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
static inline __device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
            (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                                             __longlong_as_double(assumed)));

        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif


template<typename T>
__inline__ __device__ T warpReduceSum(T val){ //TODO: benchmark, warpshuffle first then block reduction, should be faster
    for(int offset = warpSize/2;offset>0;offset/=2){
        val += __shfl_down(val,offset);
    }
    return val;
}

template<typename T>
__inline__ __device__ T BlockShuffleSum(T val){

}

template<typename ValueType, typename IndexType, int Coord, int ClusterCount, int SrcCount>
__global__ void distanceKernel1(
        ValueType* dDst /*[ClusterCount][SrcCount]*/,
        const ValueType* dData_cv,
        const IndexType* dData_ptr,
        const IndexType* dData_cid,
        const ValueType* dClusters
) {
    /** 60ms
     * tidx for one piece of data
     * tidy for one cluster
     *
     * grid_size:
     * block_size:
     * **/

    for (size_t iData = blockIdx.x * blockDim.x + threadIdx.x; iData < SrcCount; iData += gridDim.x * blockDim.x) {
        size_t end = dData_ptr[iData + 1];
        size_t p = dData_ptr[iData];
        for (size_t iCluster = blockIdx.y * blockDim.y + threadIdx.y;
             iCluster < ClusterCount; iCluster += gridDim.y * blockDim.y) {
            dDst[iData + iCluster * SrcCount] = 0;
#pragma unroll
            for (size_t iDim = 0; iDim < Coord; iDim++) {
                dDst[iData + iCluster * SrcCount] += dClusters[iCluster * Coord + iDim]*dClusters[iCluster * Coord + iDim];
                if (iDim == dData_cid[p]) {
                    dDst[iData + iCluster * SrcCount] += dData_cv[p]*dData_cv[p] - 2*dData_cv[p]*dClusters[iCluster * Coord + iDim];
                    p += (p < end - 1) ? 1 : 0;
                }
            }
        }
    }
}

template<typename ValueType, typename IndexType, int Coord, int ClusterCount, int SrcCount, int BlockSizeY, int NumCluster>
__global__ void distanceKernel11(
        ValueType* dDst /*[ClusterCount][SrcCount]*/,
        const ValueType* dData_cv,
        const IndexType* dData_ptr,
        const IndexType* dData_cid,
        const ValueType* dClusters
) {
    size_t iData = blockIdx.x * blockDim.y + threadIdx.y;

    __shared__ ValueType smCluster[NumCluster * Coord];
    __shared__ ValueType smpInter[BlockSizeY * Coord];
    __shared__ ValueType smpData[BlockSizeY * Coord];

    /** initialize **/
    for (size_t iDim = threadIdx.x; iDim < Coord; iDim += blockDim.x) {
        smpData[Coord * threadIdx.y + iDim] = (ValueType) 0;

        if (threadIdx.y < NumCluster) {
            size_t ii = NumCluster * blockIdx.y + threadIdx.y;
            if (ii < ClusterCount) {
                smCluster[threadIdx.y * Coord + iDim] = dClusters[ii * Coord + iDim];
            } else {
                smCluster[threadIdx.y * Coord + iDim] = (ValueType) 0;
            }
        }

        smpInter[Coord * threadIdx.y + iDim] = (ValueType) 0;
    }
    __syncwarp(); //TODO: benchmark, use warp sync instead


    if (iData < SrcCount) {
        /** load the data **/
        for (size_t i = dData_ptr[iData] + threadIdx.x; i < dData_ptr[iData + 1]; i += blockDim.x) {
            smpData[threadIdx.y * Coord + dData_cid[i]] = dData_cv[i];
        }
        __syncwarp();

        ValueType regData[NumCluster];
#pragma unroll
        for (size_t iiCluster = 0; iiCluster < NumCluster; iiCluster++) {
            size_t iCluster = NumCluster * blockIdx.y + iiCluster;
            for (size_t iDim = threadIdx.x; iDim < Coord; iDim += blockDim.x) {
                smpInter[threadIdx.y * Coord + iDim] =
                        (smpData[threadIdx.y * Coord + iDim] - smCluster[iiCluster * Coord + iDim]) *
                        (smpData[threadIdx.y * Coord + iDim] - smCluster[iiCluster * Coord + iDim]);
            }

            __syncwarp();

            /** reduction on the Intermediate **/ //TODO: benchmark, warp-shuffle
#pragma unroll
            for (unsigned int step = Coord / 2; step >= warpSize; step /= 2) {
                for (unsigned int iDim = threadIdx.x; iDim < step; iDim += blockDim.x) {
                    smpInter[threadIdx.y * Coord + iDim] += smpInter[threadIdx.y * Coord + iDim + step];
                }
                __syncwarp();
            }
            regData[iiCluster] = (threadIdx.x < Coord) ? smpInter[threadIdx.y * Coord + threadIdx.x] : 0;
            __syncthreads();

#pragma unroll
            for (unsigned int step = warpSize / 2; step > 0; step /= 2) {
                regData[iiCluster] += __shfl_down_sync(FULL_MASK, regData[iiCluster], step);
            }

            /** write distance back to the dDst **/
            if (threadIdx.x == 0 && iCluster < ClusterCount) {
                dDst[iData + iCluster * SrcCount] = regData[iiCluster];
            }
        }
    }
}

template<typename ValueType, typename IndexType, int Coord, int ClusterCount, int SrcCount, int BlockSizeY, int NumCluster>
__global__ void distanceKernel2(
        ValueType* dDst /*[ClusterCount][SrcCount]*/,
        const ValueType* dData_cv,
        const IndexType* dData_ptr,
        const IndexType* dData_cid,
        const ValueType* dClusters
) {
    /**
 * one block for certain pieces of data and NumCluster number of cluster
 *
 * grid_size: (ceil(SrcCount,blockDim.y), ceil(ClusterCount,NumCluster))
 * block_size:(FOLD,BLOCK_DEFAULT_SIZE/FOLD), FOLD should be warpSize
 * **/

    size_t iData = blockIdx.x * blockDim.y + threadIdx.y;

    __shared__ ValueType smCluster[NumCluster * Coord];
    __shared__ ValueType smpInter[NumCluster * BlockSizeY * Coord];
    __shared__ ValueType smpData[BlockSizeY * Coord];

    /** initialize **/
    for (size_t iDim = threadIdx.x; iDim < Coord; iDim += blockDim.x) {
        smpData[Coord * threadIdx.y + iDim] = (ValueType) 0;
        for (size_t i = 0; i < NumCluster; i++) {
            smpInter[i * BlockSizeY * Coord + Coord * threadIdx.y + iDim] = (ValueType) 0;
        }
        if (threadIdx.y < NumCluster) {
            size_t ii = NumCluster * blockIdx.y + threadIdx.y;
            if (ii < ClusterCount) {
                smCluster[threadIdx.y * Coord + iDim] = dClusters[ii * Coord + iDim];
            } else {
                smCluster[threadIdx.y * Coord + iDim] = (ValueType) 0;
            }
        }
    }
    __syncwarp(); //TODO: benchmark, use warp sync instead


    if (iData < SrcCount) {
        /** load the data **/
        for (size_t i = dData_ptr[iData] + threadIdx.x; i < dData_ptr[iData + 1]; i += blockDim.x) {
            smpData[threadIdx.y * Coord + dData_cid[i]] = dData_cv[i];
        }
        __syncwarp();

#pragma unroll
        for (size_t iiCluster = 0; iiCluster < NumCluster; iiCluster++) {
            size_t iCluster = NumCluster * blockIdx.y + iiCluster;
            for (size_t iDim = threadIdx.x; iDim < Coord; iDim += blockDim.x) {
                smpInter[iiCluster * BlockSizeY * Coord + threadIdx.y * Coord + iDim] = pow(
                        (smpData[threadIdx.y * Coord + iDim] - smCluster[iiCluster * Coord + iDim]), 2);
            }
        }
        __syncwarp();

        /** reduction on the Intermediate **/ //TODO: benchmark, warp-shuffle
        ValueType regData[NumCluster];
#pragma unroll
        for(size_t iiCluster = 0; iiCluster < NumCluster; iiCluster++) {
            for (unsigned int step = Coord / 2; step >= warpSize; step /= 2) {
                for (unsigned int iDim = threadIdx.x; iDim < step; iDim += blockDim.x) {
                    smpInter[iiCluster * BlockSizeY * Coord + threadIdx.y * Coord + iDim] += smpInter[
                            iiCluster * BlockSizeY * Coord + threadIdx.y * Coord + iDim + step];
                }
                __syncwarp();
            }
            regData[iiCluster] = (threadIdx.x < Coord) ? smpInter[iiCluster * BlockSizeY * Coord + threadIdx.y * Coord +
                                                                  threadIdx.x] : 0;

            __syncwarp();
#pragma unroll
            for (unsigned int step = warpSize / 2; step > 0; step /= 2) {
                regData[iiCluster] += __shfl_down_sync(FULL_MASK, regData[iiCluster], step);
            }

            /** write distance back to the dDst **/
            size_t iCluster = NumCluster*blockIdx.y+iiCluster;
            if (threadIdx.x == 0 && iCluster<ClusterCount) {
                dDst[iData + iCluster * SrcCount] = regData[iiCluster];
            }
        }
    }
}


template<typename ValueType, typename IndexType, int Coord, int ClusterCount, int SrcCount, int BlockSizeY>
__global__ void distanceKernel3(
        ValueType* dDst /*[ClusterCount][SrcCount]*/,
        const ValueType* dData_cv,
        const IndexType* dData_ptr,
        const IndexType* dData_cid,
        const ValueType* dClusters
) {
    /**
     * one block for certain pieces of data and a certain cluster
     *
     * grid_size: (ceil(SrcCount,blockDim.x), ClusterCount)
     * block_size:(FOLD,BLOCK_DEFAULT_SIZE/FOLD), should be warpSize
     * **/

    size_t iCluster = blockIdx.x;
    size_t iData = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ ValueType smCluster[Coord];
    __shared__ ValueType smpInter[BlockSizeY * Coord];
    /** initialize **/
    for (size_t iDim = threadIdx.x; iDim < Coord; iDim += blockDim.x) {
        if (threadIdx.y == 0) {
            smCluster[iDim] = dClusters[Coord * iCluster + iDim];
        }
        smpInter[Coord * threadIdx.y + iDim] = (ValueType) 0;
    }
    __syncthreads();

    if (iData < SrcCount) {
        for (size_t i = dData_ptr[iData] + threadIdx.x; i < dData_ptr[iData + 1]; i += blockDim.x) {
            smpInter[threadIdx.y * Coord + dData_cid[i]] = dData_cv[i];
        }
        __syncthreads();

        for (size_t iDim = threadIdx.x; iDim < Coord; iDim += blockDim.x) {
            smpInter[threadIdx.y * Coord + iDim] = (smpInter[threadIdx.y * Coord + iDim] - smCluster[iDim]) *
                                                   (smpInter[threadIdx.y * Coord + iDim] - smCluster[iDim]);
        }
        __syncthreads();

        /** reduction on the Intermediate **/ //TODO: benchmark, warp-shuffle
        for (unsigned int step = Coord / 2; step >= warpSize; step /= 2) {
            for (unsigned int iDim = threadIdx.x; iDim < step; iDim += blockDim.x) {
                smpInter[threadIdx.y * Coord + iDim] += smpInter[threadIdx.y * Coord + iDim + step];
            }
            __syncthreads();
        }
        ValueType regData = (threadIdx.x < Coord) ? smpInter[threadIdx.y * Coord + threadIdx.x] : 0;
#pragma unroll
        for (unsigned int step = warpSize / 2; step > 0; step /= 2) {
            regData += __shfl_down_sync(FULL_MASK, regData, step);
        }

        /** write distance back to the dDst **/
        if (threadIdx.x == 0) {
            dDst[iData + iCluster * SrcCount] = regData;
        }
    }
}

template<typename ValueType, typename IndexType, int Coord, int ClusterCount, int SrcCount, int BlockSizeX>
__global__ void distanceKernel31(
        ValueType* dDst /*[ClusterCount][SrcCount]*/,
        const ValueType* dData_cv,
        const IndexType* dData_ptr,
        const IndexType* dData_cid,
        const ValueType* dClusters
) {
    /** 90ms
     * one block for certain pieces of data and a certain cluster
     *
     * grid_size: (ceil(SrcCount,blockDim.x), ClusterCount)
     * block_size:(FOLD,BLOCK_DEFAULT_SIZE/FOLD), should be warpSize
     * **/

    size_t iCluster = blockIdx.x;
    size_t iData = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ ValueType smCluster[Coord];
    __shared__ ValueType smpInter[BlockSizeX * Coord];

    /** initialize **/
    for (size_t iDim = threadIdx.x; iDim < Coord; iDim += blockDim.x) {
        smpInter[Coord * threadIdx.y + iDim] = dClusters[Coord * iCluster + iDim] * dClusters[Coord * iCluster + iDim];
        if (threadIdx.y == 0) {
            smCluster[iDim] = dClusters[Coord * iCluster + iDim];
        }
    }
//    __syncthreads();
    __syncwarp();

    if (iData < SrcCount) {
        for (size_t i = dData_ptr[iData] + threadIdx.x; i < dData_ptr[iData + 1]; i += blockDim.x) {
            smpInter[threadIdx.y * Coord + dData_cid[i]] +=
                    dData_cv[i] * dData_cv[i] - 2 * smCluster[dData_cid[i]] * dData_cv[i];
        }
        //    __syncthreads();
        __syncwarp();

        /** reduction on the Intermediate **/ //TODO: benchmark, warp-shuffle
        ValueType regData = 0;
#pragma unroll_completely
        for (unsigned int iDim = threadIdx.x; iDim < Coord; iDim += warpSize) {
            regData += smpInter[threadIdx.y * Coord + iDim];
        }
        __syncwarp();
#pragma unroll_completely
        for (unsigned int step = warpSize / 2; step > 0; step /= 2) {
            regData += __shfl_down_sync(FULL_MASK, regData, step);
        }

        /** write distance back to the dDst **/
        if (threadIdx.x == 0) {
            dDst[iData + iCluster * SrcCount] = regData;
        }
    }
}

template<typename ValueType, typename IndexType, int Coord, int ClusterCount, int SrcCount, int BlockSizeX>
__global__ void distanceKernel4(
        ValueType* dDst /*[ClusterCount][SrcCount]*/,
        const ValueType* dData_cv,
        const IndexType* dData_ptr,
        const IndexType* dData_cid,
        const ValueType* dClusters
) {
    /**
     * one block for certain pieces of data and a certain cluster
     *
     * grid_size: (ceil(SrcCount,blockDim.x), ClusterCount)
     * block_size:(FOLD,BLOCK_DEFAULT_SIZE/FOLD,), should be warpSize
     * **/

    size_t iCluster = blockIdx.x;
    size_t iData = blockIdx.y * blockDim.y + threadIdx.y;
    size_t lane_id = threadIdx.x % warpSize;

    __shared__ ValueType smCluster[Coord];
    ValueType regData = 0;

    /** initialize **/
    for (size_t iDim = threadIdx.x; iDim < Coord; iDim += blockDim.x) {
        if (threadIdx.y == 0) {
            smCluster[iDim] = dClusters[Coord * iCluster + iDim];
        }
    }
    __syncthreads();

    if (iData < SrcCount) {
        IndexType p = dData_ptr[iData];
        IndexType end = dData_ptr[iData + 1];
        for (size_t iDim = threadIdx.x; iDim < Coord; iDim += blockDim.x) {
            while (p < end && dData_cid[p] < iDim)p++;
            regData += smCluster[iDim] * smCluster[iDim];
            if (dData_cid[p] == iDim) {
                regData += dData_cv[p] * dData_cv[p] - 2 * smCluster[iDim] * dData_cv[p];
            }
        }
        __syncwarp();

        /** Parallel reduction on the Intermediate, remember for small Coord and not power of 2 **/
        //TODO: benchmark, warp-shuffle, remember for Coord smaller than warpSize
#pragma unroll
        for (size_t step = warpSize / 2; step > 0; step /= 2) {
            regData += __shfl_down_sync(FULL_MASK,regData, step);
        }
        __syncwarp();
        /** write distance back to the dDst **/
        if (threadIdx.x == 0) {
            dDst[iData + iCluster * SrcCount] = regData;
        }
    }
}

template<typename ValueType, int ClusterCount, int SrcCount>
__global__ void membershipKernel1(
        const ValueType *Dst, /*[ClusterCount][SrcCount]*/
        int *Membership,
        int *dChanged
){
    unsigned int iData = threadIdx.x + blockDim.x * blockIdx.x;
    bool flag_changed = false;
    int regdChanged = *dChanged;
    if(iData < SrcCount){
        int min_cluster = 0;
        int pre_cluster = Membership[iData];
        ValueType min_dist = Dst[SrcCount*min_cluster + iData];
        for(unsigned int iCluster = 1; iCluster < ClusterCount; iCluster++){
            if(Dst[SrcCount*iCluster + iData] < min_dist){
                min_cluster = iCluster;
                min_dist = Dst[SrcCount*iCluster + iData];
            }
        }
        if(min_cluster!=pre_cluster) {
            Membership[iData] = min_cluster;
            flag_changed  = true;
        }
        if(flag_changed&&(regdChanged==0)){
            *dChanged = 1;
        }
    }
}

template<typename ValueType, typename IndexType, int Coord, int ClusterCount, int SrcCount>
inline void updateMembership(
        ValueType *dDst, /*[ClusterCount][SrcCount]*/
        int *dMembership, /*[SrcCount]*/
        int *dChanged,
        const ValueType *dData_cv,
        const IndexType *dData_ptr,
        const IndexType *dData_cid,
        const ValueType *dClusters,
        dist_kernel_t dkt,
        cudaStream_t stm=0
) {
    /** Calculate Dst **/
    if (dkt == dst1) {
        dim3 grid(GRID_DEFAULT_SIZE, 16, 1);
        dim3 block(BLOCK_DEFAULT_SIZE, 2, 1);
        distanceKernel1<ValueType, IndexType, Coord, ClusterCount, SrcCount><<<grid, block, 0, stm>>>(dDst, dData_cv,
                                                                                                      dData_ptr,
                                                                                                      dData_cid,
                                                                                                      dClusters);
    } else if (dkt == dst2) {
        // do not use
        constexpr int NumCluster = 2;
        constexpr int BlockSizeY = 2;
        dim3 block(FOLD, BlockSizeY);
        dim3 grid((SrcCount + block.y - 1) / block.y, (ClusterCount + NumCluster - 1) / NumCluster);
        distanceKernel11<ValueType, IndexType, Coord, ClusterCount, SrcCount,
                BlockSizeY, NumCluster><<<grid, block,
        (BlockSizeY * Coord + NumCluster * Coord + BlockSizeY * Coord) * sizeof(ValueType), stm>>>(dDst,
                                                                                                   dData_cv,
                                                                                                   dData_ptr,
                                                                                                   dData_cid,
                                                                                                   dClusters);

    } else if (dkt == dst3) {
        dim3 block(FOLD, BLOCK_DEFAULT_SIZE / FOLD);
        dim3 grid(ClusterCount, (SrcCount + block.y - 1) / block.y);
        distanceKernel3<ValueType, IndexType, Coord, ClusterCount, SrcCount,
                BLOCK_DEFAULT_SIZE / FOLD><<<grid, block, (Coord + BLOCK_DEFAULT_SIZE / FOLD) *
                                                          sizeof(ValueType), stm>>>(dDst, dData_cv, dData_ptr,
                                                                                    dData_cid, dClusters);
    } else if (dkt == dst31) {
        dim3 block(FOLD, BLOCK_DEFAULT_SIZE / FOLD);
        dim3 grid(ClusterCount, (SrcCount + block.y - 1) / block.y);
        distanceKernel31<ValueType, IndexType, Coord, ClusterCount, SrcCount,
                BLOCK_DEFAULT_SIZE / FOLD><<<grid, block, (Coord + BLOCK_DEFAULT_SIZE / FOLD) *
                                                          sizeof(ValueType), stm>>>(dDst, dData_cv, dData_ptr,
                                                                                    dData_cid, dClusters);

    } else if (dkt == dst4) {
        dim3 block(FOLD, BLOCK_DEFAULT_SIZE / FOLD);
        dim3 grid(ClusterCount, (SrcCount + block.x - 1) / block.y);
        distanceKernel4<ValueType, IndexType, Coord, ClusterCount, SrcCount,
                BLOCK_DEFAULT_SIZE / FOLD><<<grid, block, Coord * sizeof(ValueType), stm>>>(dDst, dData_cv, dData_ptr,
                                                                                            dData_cid, dClusters);
    }

    checkCuda();

    /** Update Membership **/
    dim3 grid2((SrcCount + BLOCK_DEFAULT_SIZE - 1) / BLOCK_DEFAULT_SIZE), block2(BLOCK_DEFAULT_SIZE);
    membershipKernel1<ValueType, ClusterCount, SrcCount><<<grid2, block2, 0, stm>>>(dDst, dMembership, dChanged);
    checkCuda();
    checkCuda(cudaDeviceSynchronize());

}

template<typename ValueType, typename IndexType, int Coord, int ClusterCount, int SrcCount>
void updateClustersMemberCount_cpu(
        ValueType *Clusters, /*[Coord][ClusterCount]*/
        int *MemberCount,
        const int *Membership,
        const ValueType *Data_cv,
        const IndexType *Data_ptr,
        const IndexType *Data_cid
){
    std::fill(Clusters,Clusters+Coord*ClusterCount,0);
    std::fill(MemberCount,MemberCount+ClusterCount,0);
    for(unsigned int i = 0; i < SrcCount; i++){
        MemberCount[Membership[i]] += 1;
        for(unsigned int j = Data_ptr[i]; j < Data_ptr[i+1]; j++){
            Clusters[ClusterCount*Data_cid[j]+Membership[i]] += Data_cv[j];
        }
    }

    for(unsigned int j = 0; j < ClusterCount; j++){
        if(MemberCount[j]!=0) {
            for (unsigned int idim = 0; idim < Coord; idim++) {
                Clusters[ClusterCount * idim + j] /= MemberCount[j];
            }
        }
        else{
//            for (unsigned int idim = 0; idim < Coord; idim++) {
//                Clusters[ClusterCount * idim + j] = (ValueType)idim;
//            }
        }
    }
}

template<int SrcCount>
__global__ void updateMemberCountKernel(const int *dMembership, int *dMemberCount) {
    /**
     * one block for one kind of cluster, load data in the shared memory and then sum them up
     * no need to initialize the MemberCount
     *
     * grid_size: ClusterCount
     * block_size: BLOCK_DEFAULT_SIZE
     * **/

    unsigned int iCluster = blockIdx.x;
    unsigned int tid = threadIdx.x;
    __shared__ int sm[SrcCount]; //FIXME: for extreme large SrcCount?
    for (unsigned int i = tid; i < SrcCount; i += blockDim.x) {
        sm[i] = (int) (dMembership[i] == iCluster);
    }
    __syncthreads();

    /** perform parallel reduction **/
    for (unsigned int step = SrcCount / 2; step >= warpSize; step /= 2) {
        __syncthreads();
        if (tid + step < SrcCount) {
            sm[tid] += sm[tid + step];
        }
    }
    int reg_data = sm[tid];
#pragma unroll
    for (unsigned int step = warpSize / 2; step > 0; step /= 2) {
        reg_data += __shfl_xor_sync(0xffffffff, reg_data, step, warpSize);
    }

    /** write back to MemberCount **/
    if (threadIdx.x == 0) {
        if (SrcCount % 2 == 1) {
            reg_data += sm[SrcCount - 1];
        }
        dMemberCount[iCluster] = reg_data;
    }
}

template<int SrcCount,int blockDim,int ClusterCount>
__global__ void updateMemberCountKernel2(const int *dMembership, int *dMemberCount) {
    /**
     * for extreme large SrcCount
     * need to initialize dMemberCount first
     *
     * grid_size: (ceil(SrcCount/blockDim),CLusterCount) //TODO: set max number of grid_size.x
     * block_size: BLOCK_DEFAULT_SIZE
     * **/

    __shared__ int sm[blockDim];
    int iCluster = blockIdx.y;
    int iData = blockDim * blockIdx.x + threadIdx.x;

    sm[threadIdx.x] = (iData < SrcCount) ? (int) (dMembership[iData] == iCluster)
                                         : 0; //TODO: benchmark, it should be better than if-else switch?

    //TODO: benchmark, warp-shuffle first reduction
    for (size_t step = blockDim / 2; step > 0; step /= 2) {
        __syncthreads();
        if(threadIdx.x<step) {
            sm[threadIdx.x] += sm[threadIdx.x + step];
        }
    }

    int reg_data = sm[threadIdx.x];
//#pragma unroll
//    for (size_t step = warpSize / 2; step > 0; step /= 2) {
//        if(threadIdx.x<step) {
//            reg_data += __shfl_xor_sync(0xffffffff, reg_data, step, warpSize);
//        }
//    }

    if (threadIdx.x == 0) {
        atomicAdd(&dMemberCount[iCluster], reg_data);
    }
}

template<int ClusterCount, int SrcCount>
inline void updateMemberCount(const int *dMembership, int *dMemberCount,cudaStream_t stm=0) { //TODO:benchmark for 2 kind of computation

//        dim3 grid(ClusterCount);
//        dim3 block(SrcCount);
//        updateMemberCountKernel<SrcCount><<<grid, block>>>(dMembership, dMemberCount);

    dim3 grid((SrcCount + BLOCK_DEFAULT_SIZE - 1) / BLOCK_DEFAULT_SIZE, ClusterCount);
    dim3 block(BLOCK_DEFAULT_SIZE);
    updateMemberCountKernel2<SrcCount, BLOCK_DEFAULT_SIZE, ClusterCount><<<grid, block, BLOCK_DEFAULT_SIZE*sizeof(int),stm>>>(dMembership,dMemberCount);
}

template<typename ValueType, typename IndexType, int Coord, int ClusterCount, int SrcCount>
__global__ void updateClustersKernel_naive(
        ValueType *dClusters, /*[kClusterCount][kCoord]*/
        const int *dMembership, /*[kSrcCount]*/
        const int *dMemberCount,
        const ValueType *dData_cv,
        const IndexType *dData_ptr,
        const IndexType *dData_cid
) {
    /**
     * naive GPU version, one line of threads for one cluster, used as reference
     *
     * grid_size: 1
     * block_size: (kClusterCount,64)
     **/
    unsigned int iCluster = threadIdx.x;
    int regMemberNum = dMemberCount[iCluster];
    if (regMemberNum != 0) {
        for (unsigned int i = 0; i < SrcCount; i++) {
            if (dMembership[i] == iCluster) {
                for (IndexType j = dData_ptr[i] + threadIdx.y; j < dData_ptr[i + 1]; j += blockDim.y) {
                    atomicAdd(&dClusters[iCluster * Coord + dData_cid[j]], dData_cv[j] / (ValueType) regMemberNum);
                }
            }
        }
    }
}

template<typename ValueType, typename IndexType, int Coord, int ClusterCount, int SrcCount>
__global__ void updateClustersKernel_SharedMemory(
        ValueType *dClusters, /*[kClusterCount][kCoord]*/
        const int *dMembership, /*[kSrcCount]*/
        const int *dMemberCount,
        const ValueType *dData_cv,
        const IndexType *dData_ptr,
        const IndexType *dData_cid
) {
    /**
     * one block update one clusters, with atomic lock write-back
     *
     * grid_size: ClusterCount
     * block_size: BLOCK_DEFAULT_SIZE
     * **/
    __shared__ ValueType smCluster[Coord]; //here we store the datas that belong to one cluster, and reduction later
    unsigned int block_size = blockDim.x;
    unsigned int iCluster = blockIdx.x;
    int regMemberNum = dMemberCount[iCluster];
    if (regMemberNum != 0) {
        /** initialize the shared memory **/
        for (unsigned int i = threadIdx.x; i < Coord; i += block_size) {
            smCluster[i] = (ValueType) 0;
        }
        __syncthreads();

        /** sum up with atomic lock **/
        for (unsigned int i = threadIdx.x; i < SrcCount; i += block_size) {
            if (dMembership[i] == iCluster) {
                for (IndexType j = dData_ptr[i]; j < dData_ptr[i + 1]; j++) {
//                smCluster[dData_cid[j]] += dData_cv[j]/regMemberNum; //atomic add
                    atomicAdd(&smCluster[dData_cid[j]], dData_cv[j]);
                }
            }
        }
        __syncthreads();

        /** write back **/
        for (unsigned int i = threadIdx.x; i < Coord; i += block_size) {
            dClusters[iCluster * Coord + i] = smCluster[i] / (ValueType) regMemberNum;
        }
    } else {
//        for (unsigned int i = threadIdx.x; i < Coord; i += block_size) {
//            dClusters[iCluster * Coord + i] = (ValueType)i;
//        }
    }
}

template<typename ValueType, typename IndexType, int Coord, int ClusterCount, int SrcCount, int BlockSize>
__global__ void updateClustersKernel_SharedMemory2(
        ValueType *dClusters, /*[kClusterCount][kCoord]*/
        const int *dMembership, /*[kSrcCount]*/
        const int *dMemberCount,
        const ValueType *dData_cv,
        const IndexType *dData_ptr,
        const IndexType *dData_cid
) {
    /**
     * one block update one clusters, with atomic lock write-back
     *
     * grid_size: ClusterCount,
     * block_size: BLOCK_DEFAULT_SIZE
     * **/
    __shared__ ValueType smCluster[Coord]; //here we store the datas that belong to one cluster, and reduction later
    unsigned int iCluster = blockIdx.x;
    int regMemberNum = dMemberCount[iCluster];
    if (regMemberNum != 0) {
        /** initialize the shared memory **/
        for (unsigned int i = threadIdx.x; i < Coord; i += BlockSize) {
            smCluster[i] = (ValueType) 0;
        }
        /** sum up with atomic lock **/
        for (unsigned int i = threadIdx.x; i < SrcCount; i += BlockSize) {
            if (dMembership[i] == iCluster) {
                for (IndexType j = dData_ptr[i]; j < dData_ptr[i + 1]; j++) {
                    atomicAdd(&smCluster[dData_cid[j]], dData_cv[j]);
                }
            }
        }
        __syncthreads();
        /** write back **/
        for (unsigned int i = threadIdx.x; i < Coord; i += BlockSize) {
            dClusters[iCluster * Coord + i] = smCluster[i] / (ValueType) regMemberNum;
        }
    }
}


template<typename ValueType, typename IndexType, int Coord, int ClusterCount, int SrcCount, int BLockSizeX>
__global__ void updateClusterKernel_ParallelReduction1(
        ValueType *dClusters, /*[kClusterCount][kCoord]*/
        const int *dMembership, /*[kSrcCount]*/
        const int *dMemberCount,
        const ValueType *dData_cv,
        const IndexType *dData_ptr,
        const IndexType *dData_cid
) {
    /**
     * one block picks up data that belongs to one cluster from certain number of data, and loads them in the shared memory
     * blockDim.y represents the cluster it belongs to
     * perform parallel reduction in the shared memory //TODO: warp shuffle
     * then add up all blocks that belong to the same cluster with atomicAdd
     *
     * grid_size:(ceil(SrcCount/blockDim.x),iCluster)
     * block_size:(BLOCK_DEFAULT_SIZE/8,8)
     * **/

    unsigned int iData = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int iCluster = blockIdx.y;
    int regMemberNum = dMemberCount[iCluster];
    if (regMemberNum != 0) {
//        extern __shared__ __align__ (sizeof(ValueType)) char work[];
//        ValueType *sm = reinterpret_cast<ValueType *>(work); /*[block_size][Coord]*/
        __shared__ ValueType sm[BLockSizeX * Coord];
        for (unsigned int i = threadIdx.x * blockDim.y + threadIdx.y;
             i < blockDim.x * Coord; i += blockDim.x * blockDim.y) {
            sm[i] = 0;
        }
        __syncthreads();

        if (iData < SrcCount) {
            if (dMembership[iData] == iCluster) {
                for (unsigned int i = dData_ptr[iData] + threadIdx.y; i < dData_ptr[iData + 1]; i += blockDim.y) {
                    sm[iData * Coord + dData_cid[i]] = dData_cv[i];
                }
            }
        }
        __syncthreads();

        /** parallel reduction within the block, without warp shuffle **/
        //TODO: warp shuffle
        for (unsigned int step = blockDim.x / 2; step > 0; step /= 2) {
            __syncthreads();
            if (threadIdx.x < step) {
                for (unsigned int iDim = threadIdx.y; iDim < Coord; iDim += blockDim.y) {
                    sm[Coord * iData + iDim] += sm[Coord * (iData + step) + iDim];
                }
            }
        }
        __syncthreads();

        /** write back to dClusters with atomic lock **/
        if (threadIdx.x == 0) {
            for (unsigned int iDim = threadIdx.y; iDim < Coord; iDim += blockDim.y) {
                atomicAdd(&dClusters[Coord * iCluster + iDim], sm[iDim] / (ValueType) regMemberNum);
            }
        }
    } else {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            for (unsigned int iDim = threadIdx.y; iDim < Coord; iDim += blockDim.y) {
                dClusters[Coord * iCluster + iDim] = (ValueType) iDim - iCluster;
            }
        }
    }
}


template<typename ValueType, typename IndexType, unsigned int InterDim2, int Coord, int ClusterCount, int SrcCount>
__global__ void updateClusterKernel_MoreParallelReduction_step1(
        ValueType *dIntermediate, /*[kClusterCount][ceil(SrcCount/BLOCK_SIZE.x)][kCoord]*/
        const int *dMembership, /*[kSrcCount]*/
        const int *dMemberCount,
        const ValueType *dData_cv,
        const IndexType *dData_ptr,
        const IndexType *dData_cid
) {
    /**
     * use intermedia variable to store the reduced result from all blocks,
     * later perform reduction on them (in step2 kernel)
     *
     * grid_size:(ceil(SrcCount/BlockSize.x),iCluster)
     * block_size:(BLOCK_DEFAULT_SIZE/8,8)
     * **/

    unsigned int iData = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int iCluster = blockIdx.y;
//    int regMemberNum = dMemberCount[iCluster];
//    unsigned int start = iCluster * InterDim2 * Coord;

    extern __shared__ __align__ (sizeof(ValueType)) char work[];
    ValueType *sm = reinterpret_cast<ValueType *>(work); /*[block_size.x][Coord]*/
    for (unsigned int iDim = threadIdx.y; iDim < Coord; iDim += blockDim.y) {
        sm[Coord * threadIdx.x + iDim] = 0;
    }
    __syncthreads(); //TODO: benchmark, use atomic instead of syncthreads

    if (iData < SrcCount) {
        if (dMembership[iData] == iCluster) {
            for (unsigned int i = dData_ptr[iData] + threadIdx.y; i < dData_ptr[iData + 1]; i += blockDim.y) {
                sm[Coord * threadIdx.x + dData_cid[i]] = dData_cv[i];
            }
        }
    }
    __syncthreads();

    /** parallel reduction within the block, without warp shuffle **/
    //TODO：warp shuffle here
    for (unsigned int step = blockDim.x / 2; step > 0; step /= 2) {
        __syncthreads();
        if (threadIdx.x < step) {
            for (unsigned int iDim = threadIdx.y; iDim < Coord; iDim += blockDim.y) {
                sm[Coord * threadIdx.x + iDim] += sm[Coord * (threadIdx.x + step) + iDim];
            }
        }
    }
    __syncthreads();

    /** write result in the shared memory back to the intermediate **/
    if (threadIdx.x == 0) {
        for (unsigned int iDim = threadIdx.y; iDim < Coord; iDim += blockDim.y) {
            dIntermediate[iCluster * InterDim2 * Coord + blockIdx.x * Coord + iDim] = sm[iDim]; //FIXME: for odd number of blocks
        }
    }
    __syncthreads();

}

template<typename ValueType, typename IndexType, int Coord, int ClusterCount, int InterDim2, int BlockSizeX>
__global__ void updateClusterKernel_MoreParallelReduction_step2_2(
        ValueType *dIntermediate, /*[kClusterCount][InterDim2][kCoord]*/
        ValueType *dClusters, /*[kClusterCount][kCoord]*/
        const int *dMemberCount
) {
    /**
     * blockId.y for cluster, load the Intermediate to shared memory,
     * optimized for large SrcCount
     *
     * grid_size: (ceil(InterDim2/block_size.x),CLusterCount)
     * block_size: (BLOCK_DEFAULT_SIZE/FOLD,FOLD)
     * **/

    unsigned int iCluster = blockIdx.y;
    unsigned int iiData = BlockSizeX * blockIdx.x + threadIdx.x;
    unsigned int Start = iCluster * InterDim2 * Coord;

    int regMemberCount = dMemberCount[iCluster];
    __shared__ ValueType sm[BlockSizeX * Coord];

    for (size_t iDim = threadIdx.y; iDim < Coord; iDim += blockDim.y) {
        sm[threadIdx.x * Coord + iDim] = (iiData<InterDim2)? dIntermediate[Start + iiData * Coord + iDim]:0; //TODO:benchmark, should be better than if-else
    }
    __syncthreads();

    for (size_t step = BlockSizeX / 2; step > 0; step /= 2) { //TODO: warp-shuffle reduction
        __syncthreads();
        if (threadIdx.x < step) {
            for (size_t iDim = threadIdx.y; iDim < Coord; iDim += blockDim.y) {
                sm[threadIdx.x * Coord + iDim] += sm[(threadIdx.x + step) * Coord + iDim];
            }
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        for (size_t iDim = threadIdx.y; iDim < Coord; iDim += blockDim.y) {
            ValueType tmp = sm[iDim];
            if (regMemberCount != 0) {
                atomicAdd(&dClusters[iCluster * Coord + iDim], tmp / (ValueType) regMemberCount);
            }
        }
    }
}

template<typename ValueType, typename IndexType, int Coord, int ClusterCount, int InterDim2, int BlockSizeX>
__global__ void updateClusterKernel_MoreParallelReduction_step2_3(
        ValueType *dIntermediate, /*[kClusterCount][InterDim2][kCoord]*/
        ValueType *dClusters, /*[kClusterCount][kCoord]*/
        const int *dMemberCount
){

}

template<typename ValueType, typename IndexType, int Coord, int ClusterCount, int SrcCount>
int updateClusters_cuda(
        cluster_kernel_t kernel_type,
        ValueType *dClusters,
        ValueType *dIntermediate,
        const int *dMembership, /*[kSrcCount]*/
        const int *dMemberCount, /*[kClusterCount]*/
        const ValueType *dData_cv,
        const IndexType *dData_ptr,
        const IndexType *dData_cid,
        cudaStream_t stm = 0

) {
    if (kernel_type == naive) {
        dim3 block_size(ClusterCount, 64);
        updateClustersKernel_naive<ValueType, IndexType, Coord, ClusterCount, SrcCount><<<1, block_size,0,stm>>>(
                dClusters,
                dMembership,
                dMemberCount,
                dData_cv,
                dData_ptr,
                dData_cid);
        checkCuda();
    } else if (kernel_type == SharedMemory) {
        dim3 block_size(BLOCK_DEFAULT_SIZE);
        updateClustersKernel_SharedMemory<ValueType, IndexType, Coord, ClusterCount, SrcCount><<<ClusterCount, block_size,sizeof(ValueType)*Coord,stm>>>(
                dClusters, dMembership, dMemberCount, dData_cv, dData_ptr, dData_cid);
        checkCuda();
    } else if (kernel_type == SharedMemory2) {
        constexpr int BlockSize = 32;
        dim3 block_size(BlockSize);
        updateClustersKernel_SharedMemory2<ValueType, IndexType, Coord, ClusterCount, SrcCount, BlockSize><<<ClusterCount, block_size,sizeof(ValueType)*Coord,stm>>>(
                dClusters, dMembership, dMemberCount, dData_cv, dData_ptr, dData_cid);
        checkCuda();
    } else if (kernel_type == ParallelReduction) {
        dim3 block_size(BLOCK_DEFAULT_SIZE / 8, 8);
        dim3 grid_size((SrcCount + block_size.x - 1) / block_size.x, ClusterCount);
//        updateClusterKernel_ParallelReduction1<ValueType, IndexType, Coord, ClusterCount, SrcCount,
//                BLOCK_DEFAULT_SIZE / 8><<<grid_size, block_size>>>(dClusters, dMembership, dMemberCount, dData_cv,
//                                                                   dData_ptr, dData_cid);
        checkCuda();
    } else if (kernel_type == MoreParallelReduction) {
        constexpr int InterDim2 = (SrcCount + BLOCK_DEFAULT_SIZE / FOLD - 1) / (BLOCK_DEFAULT_SIZE / FOLD); //TODO: limit block number
        dim3 block_size(BLOCK_DEFAULT_SIZE / FOLD, FOLD);
        dim3 grid_size(InterDim2, ClusterCount);
        updateClusterKernel_MoreParallelReduction_step1<ValueType, IndexType, InterDim2, Coord, ClusterCount, SrcCount><<<grid_size, block_size,
        block_size.x * Coord * sizeof(ValueType)>>>(dIntermediate, dMembership, dMemberCount, dData_cv, dData_ptr, dData_cid);
        checkCuda();

        constexpr int BlockSizeX = BLOCK_DEFAULT_SIZE / FOLD;
        dim3 grid_size2((InterDim2 + BlockSizeX - 1) / BlockSizeX, ClusterCount);
        dim3 block_size2(BlockSizeX, FOLD);

        updateClusterKernel_MoreParallelReduction_step2_2<ValueType, IndexType, Coord, ClusterCount, InterDim2, BlockSizeX><<<grid_size2, block_size2>>>(
                dIntermediate, dClusters, dMemberCount);
    }

    return 0;
}

template<typename ValueType, typename IndexType, int SrcCount, int ClusterCount, int NumNz>
int unittest_ClusterUpdate_MembershipUpdate(std::string path, cluster_kernel_t ct) {
    int num_nz;
    get_nz<ValueType>(path, num_nz);

    ValueType Data_cv[100];
    IndexType Data_ptr[31];
    IndexType Data_cid[100];
    read_csr<ValueType, IndexType>(path, Data_cv, Data_ptr, Data_cid);

    ValueType Clusters[30]{0};

    int Membership[30]{0};
    read_Membership(path + "tcsr_membership.txt", Membership);
    printmat(Membership, 1, 30, __LINE__);

//    std::ifstream fMembership(path,std::ifstream::in);
//    ValueType tmp;
//    for(int i = 0;fMembership>>tmp;i++){
//        refMembership[i] = tmp;
//    }

    int MemberCount[3]{10, 10, 10};

    ValueType *dData_cv;
    ValueType *dClusters;
    ValueType *dIntermediate;
    IndexType *dData_ptr;
    IndexType *dData_cid;
    int *dMembership;
    int *dMemberCount;

    checkCuda(cudaMalloc((void **) &dData_cv, 100 * sizeof(ValueType)));
    checkCuda(cudaMalloc((void **) &dData_cid, 100 * sizeof(IndexType)));
    checkCuda(cudaMalloc((void **) &dData_ptr, 31 * sizeof(IndexType)));
    checkCuda(cudaMalloc((void **) &dClusters, 3 * 10 * sizeof(ValueType)));
    checkCuda(cudaMalloc((void **) &dMembership, 30 * sizeof(int)));
    checkCuda(cudaMalloc((void **) &dMemberCount, 3 * sizeof(int)));

    if (ct == MoreParallelReduction) {
        dim3 block_size(BLOCK_DEFAULT_SIZE / 8, 8);
        dim3 grid_size((30 + block_size.x - 1) / block_size.x, 3);
        checkCuda(cudaMalloc((void **) &dIntermediate, 3 * grid_size.x * 10 * sizeof(ValueType)));
    }

    checkCuda(cudaMemcpy(dData_cv, Data_cv, 100 * sizeof(ValueType), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(dData_ptr, Data_ptr, 31 * sizeof(IndexType), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(dData_cid, Data_cid, 100 * sizeof(IndexType), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(dMembership, Membership, 30 * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(dMemberCount, MemberCount, 3 * sizeof(int), cudaMemcpyHostToDevice));

    /** test ClusterUpdate **/
    updateClusters_cuda<ValueType, IndexType, 10, 3, 30>(ct, dClusters, dIntermediate, dMembership, dMemberCount,
                                                         dData_cv, dData_ptr, dData_cid,0);

    checkCuda(cudaMemcpy(Clusters, dClusters, 30 * sizeof(ValueType), cudaMemcpyDeviceToHost));
    printmat(Clusters, 3, 10, __LINE__);

    /** test MembershipUpdate (and Dst) **/
    std::fill(Membership,Membership+30,0);
    checkCuda(cudaMemcpy(dMembership,Membership,30*sizeof(int),cudaMemcpyHostToDevice));

    int *dChanged;
    checkCuda(cudaMalloc((void**)&dChanged,1*sizeof(int)));
    checkCuda(cudaMemset(dChanged,0,1*sizeof(int)));

    ValueType *dDst;
    checkCuda(cudaMalloc((void**)&dDst,30*3*sizeof(ValueType)));
    checkCuda(cudaMemset(dDst,0,30*3*sizeof(ValueType)));
    updateMembership<ValueType,IndexType,10,3,30>(dDst,dMembership,dChanged,dData_cv,dData_ptr,dData_cid,dClusters);
    checkCuda(cudaMemcpy(Membership,dMembership,30*sizeof(int),cudaMemcpyDeviceToHost));
    printmat(Membership,1,30,__LINE__);

    /** test MemberCountUpdate **/
    checkCuda(cudaMemset(dMemberCount,0,3*sizeof(int)));
    updateMemberCount<3,30>(dMembership,dMemberCount);
    checkCuda(cudaMemcpy(MemberCount,dMemberCount,3*sizeof(int),cudaMemcpyDeviceToHost));
    printmat(MemberCount,1,3,__LINE__);

    cudaFree(dData_cv);
    cudaFree(dData_ptr);
    cudaFree(dData_cid);
    cudaFree(dMembership);
    if(ct==MoreParallelReduction){
        cudaFree(dIntermediate);
    }

    return 0;
}

template<typename ValueType, typename IndexType, int Coord, int ClusterCount, int SrcCount>
void ClusterInit(ValueType *Clusters, ValueType *Data_cv, IndexType *Data_ptr, IndexType *Data_col) {
    const size_t Step = SrcCount / ClusterCount;
    std::fill(Clusters, Clusters + Coord * ClusterCount, 0);
    size_t sample = (rand()+1) % Step;
    for (size_t i = 0; i < ClusterCount; i++) {
        sample += (rand()+1) % Step;
        for (size_t j = Data_ptr[sample]; j < Data_ptr[sample + 1]; j++) {
            Clusters[i * Coord + Data_col[j]] = Data_cv[j];
        }
    }
}

template<typename ValueType, typename IndexType, int kCoords, int kClusterCount, int kSrcCount, int LoopIteration,int WaitForChange=10>
int CallfuncSync(std::string path, cluster_kernel_t ckt=SharedMemory, dist_kernel_t dkt=dst1) {
    ValueType *Data_cv;
    IndexType *Data_ptr;
    IndexType *Data_cid;
    int *Membership;
    int *MemberCount;
    ValueType *Clusters;
    int Changed = 0;
    int SumNoChanged = 0;

    int num_nz;
    get_nz<ValueType>(path, num_nz);
    if (num_nz <= 0) {
        std::cout << "Failed to read the data!" << std::endl;
        return -1;
    } else {
        std::cout << "get the number of non-zero elements:"<<num_nz<<std::endl<<std::endl;
    }

    checkCuda(cudaHostAlloc((void **) &Data_cv, num_nz * sizeof(ValueType), cudaHostAllocDefault));
    checkCuda(cudaHostAlloc((void **) &Data_ptr, (kSrcCount + 1) * sizeof(IndexType), cudaHostAllocDefault));
    checkCuda(cudaHostAlloc((void **) &Data_cid, num_nz * sizeof(IndexType), cudaHostAllocDefault));

    read_csr<ValueType, IndexType>(path, Data_cv, Data_ptr, Data_cid);

    checkCuda(cudaHostAlloc((void **) &Clusters, kClusterCount * kCoords * sizeof(ValueType),
                            cudaHostAllocDefault)); //FIXME: Initialize Cluster Center?
//    for (unsigned int i = 0; i < kClusterCount * kCoords; i++) { Clusters[i] = rand()%10; }
    ClusterInit<ValueType,IndexType,kCoords,kClusterCount,kSrcCount>(Clusters,Data_cv,Data_ptr,Data_cid);
    printmat(Clusters,kClusterCount,kCoords,__LINE__);

    checkCuda(cudaHostAlloc((void **) &Membership, kSrcCount * sizeof(int),
                            cudaHostAllocDefault)); //use fixed host memory
    std::fill(Membership, Membership + kSrcCount, 0);

    checkCuda(cudaHostAlloc((void **) &MemberCount, kClusterCount * sizeof(int), cudaHostAllocDefault));
    std::fill(MemberCount, MemberCount + kClusterCount, 0);

    ValueType *Dst;
    checkCuda(cudaHostAlloc((void**)&Dst,kSrcCount*kClusterCount*sizeof(ValueType),cudaHostAllocDefault));

    ValueType *dData_cv;
    IndexType *dData_ptr;
    IndexType *dData_cid;
    int *dMembership;
    int *dMemberCount;
    ValueType *dClusters;
    int *dChanged;
    ValueType *dDst;
    ValueType *dIntermediate;

    checkCuda(cudaMalloc((void **) &dData_cv, num_nz * sizeof(ValueType)));
    checkCuda(cudaMalloc((void **) &dData_ptr, (kSrcCount + 1) * sizeof(IndexType)));
    checkCuda(cudaMalloc((void **) &dData_cid, num_nz * sizeof(IndexType)));

    checkCuda(cudaMalloc((void **) &dClusters, kClusterCount * kCoords * sizeof(ValueType)));
    checkCuda(cudaMalloc((void **) &dMembership, kSrcCount * sizeof(int)));
    checkCuda(cudaMalloc((void **) &dMemberCount, kClusterCount * sizeof(int)));
    checkCuda(cudaMalloc((void **) &dChanged, 1 * sizeof(int)));
    checkCuda(cudaMalloc((void **) &dDst, kClusterCount * kSrcCount * sizeof(ValueType)));

    if (ckt == MoreParallelReduction) {
        dim3 block_size(BLOCK_DEFAULT_SIZE / FOLD, FOLD);
        dim3 grid_size((kSrcCount + block_size.x - 1) / block_size.x, kClusterCount);
        checkCuda(cudaMalloc((void **) &dIntermediate, kClusterCount * grid_size.x * kCoords * sizeof(ValueType)));
        checkCuda(cudaMemset(dIntermediate,0,kClusterCount * grid_size.x * kCoords * sizeof(ValueType)));
    }

    checkCuda(cudaMemcpy(dData_cv, Data_cv, num_nz * sizeof(ValueType), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(dData_ptr, Data_ptr, (kSrcCount + 1) * sizeof(IndexType), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(dData_cid, Data_cid, num_nz * sizeof(IndexType), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(dMembership, Membership, kSrcCount * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(dMemberCount, MemberCount, kClusterCount * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(dClusters, Clusters, kCoords * kClusterCount * sizeof(ValueType), cudaMemcpyHostToDevice));

    int itCount = 0;
    do {
        /** update Memberships **/
        checkCuda(cudaMemset(dChanged,0,sizeof(int)));
        updateMembership<ValueType, IndexType, kCoords, kClusterCount, kSrcCount>(dDst, dMembership, dChanged, dData_cv,
                                                                                  dData_ptr, dData_cid, dClusters,dkt,0);
        checkCuda();
#if defined(DEBUG)
        checkCuda(cudaMemcpy(Membership, dMembership, kSrcCount * sizeof(int), cudaMemcpyDeviceToHost));
        printmat(Membership,1,kSrcCount,__LINE__);
//        checkCuda(cudaMemcpy(Dst,dDst,kSrcCount*kClusterCount*sizeof(ValueType),cudaMemcpyDeviceToHost));
//        printmat(Dst,kClusterCount,kSrcCount,__LINE__);
#endif

        /** update MemberCount **/
        checkCuda(cudaMemset(dMemberCount,0,kClusterCount*sizeof(int)));
        updateMemberCount<kClusterCount, kSrcCount>(dMembership, dMemberCount);
        checkCuda();
#if defined(DEBUG)
        checkCuda(cudaMemcpy(MemberCount, dMemberCount, kClusterCount * sizeof(int), cudaMemcpyDeviceToHost));
        printmat(MemberCount, 1, kClusterCount, __LINE__);
#endif

        /** update Clusters **/
        if (ckt == cpu) {
            checkCuda(cudaMemcpy(Membership, dMembership, kSrcCount * sizeof(int), cudaMemcpyDeviceToHost));
            updateClustersMemberCount_cpu<ValueType, IndexType, kCoords, kClusterCount, kSrcCount>(Clusters,
                                                                                                   MemberCount,
                                                                                                   Membership, Data_cv,
                                                                                                   Data_ptr,
                                                                                                   Data_cid);
            checkCuda();
            checkCuda(cudaMemcpy(dClusters, Clusters, kCoords * kClusterCount * sizeof(ValueType),
                                 cudaMemcpyHostToDevice));

        } else {
            checkCuda(cudaMemset(dClusters,0,kClusterCount*kCoords*sizeof(ValueType)));
            updateClusters_cuda<ValueType, IndexType, kCoords, kClusterCount, kSrcCount>(ckt, dClusters, dIntermediate,
                                                                                         dMembership, dMemberCount,
                                                                                         dData_cv, dData_ptr,
                                                                                         dData_cid); //TODO cudaStream as parameters
#if defined(DEBUG)
            checkCuda(cudaMemcpy(Clusters, dClusters, kCoords * kClusterCount * sizeof(ValueType),
                                 cudaMemcpyDeviceToHost));
            printmat(Clusters, kClusterCount, kCoords, __LINE__);
#endif
        }
        checkCuda(cudaMemcpy(&Changed, dChanged, sizeof(int), cudaMemcpyDeviceToHost));
        SumNoChanged = (Changed == 0) ? (SumNoChanged + 1) : 0;
        printf(">>>>>>>>>>>loop %d/%d \n",itCount,LoopIteration);
    } while ((SumNoChanged < WaitForChange) && (itCount++ < LoopIteration));

    std::cout << "it count: " << itCount << std::endl;

    cudaFree(dData_cv);
    cudaFree(dData_ptr);
    cudaFree(dData_cid);
    cudaFree(dMembership);
    cudaFree(dChanged);
    cudaFree(dDst);
    if(ckt == MoreParallelReduction){
        cudaFree(dIntermediate);
    }

    cudaFreeHost(Data_cv);
    cudaFreeHost(Data_ptr);
    cudaFree(Data_cid);
    cudaFreeHost(Membership);
    cudaFreeHost(MemberCount);
    cudaFreeHost(Clusters);

    return 0;
}

template<typename ValueType, typename IndexType, int kCoords, int kClusterCount, int kSrcCount, int LoopIteration=200, int WaitForChange=10>
int CallfuncSingleStream(std::string path,cluster_kernel_t ct, dist_kernel_t dkt) {
    ValueType *Data_cv;
    IndexType *Data_ptr;
    IndexType *Data_cid;
    int *Membership;
    int *MemberCount;
    ValueType *Clusters;
    int Changed = 0;
//    ValueType *Dst = new ValueType[kClusterCount * kSrcCount]{0};

    int num_nz;
    get_nz<ValueType>(path, num_nz);
    if (num_nz <= 0) {
        std::cout << "Failed to read the data!" << std::endl;
        return -1;
    } else {
        std::cout << "get the number of non-zero elements";
    }

    checkCuda(cudaHostAlloc((void **) &Data_cv, num_nz * sizeof(ValueType), cudaHostAllocDefault));
    checkCuda(cudaHostAlloc((void **) &Data_ptr, (kSrcCount + 1) * sizeof(IndexType), cudaHostAllocDefault));
    checkCuda(cudaHostAlloc((void **) &Data_cid, num_nz * sizeof(IndexType), cudaHostAllocDefault));

    read_csr<ValueType, IndexType>(path, Data_cv, Data_ptr, Data_cid);

    checkCuda(cudaHostAlloc((void **) &Clusters, kClusterCount * kCoords * sizeof(ValueType),
                            cudaHostAllocDefault)); //FIXME: Initialize Cluster Center?
//    for (unsigned int i = 0; i < kClusterCount * kCoords; i++) { Clusters[i] = rand() % 10; }
    ClusterInit<ValueType,IndexType,kCoords,kClusterCount,kSrcCount>(Clusters,Data_cv,Data_ptr,Data_cid);

    checkCuda(cudaHostAlloc((void **) &Membership, kSrcCount * sizeof(int),
                            cudaHostAllocDefault)); //use fixed host memory
    std::fill(Membership, Membership + kSrcCount, 0);

    checkCuda(cudaHostAlloc((void **) &MemberCount, kClusterCount * sizeof(int), cudaHostAllocDefault));
    std::fill(MemberCount, MemberCount + kClusterCount, 0);

    cudaStream_t stm;
//    cudaStreamCreate(&stm);
    checkCuda(cudaStreamCreateWithFlags(&stm, cudaStreamDefault)); //Blocking Stream, block the NULL stream
    cudaStreamCreateWithFlags(&stm,cudaStreamNonBlocking);
    const int EventNum = 10;
    cudaEvent_t event[EventNum];
    for (size_t i = 0; i < EventNum; i++) {
        cudaEventCreate(&event[i]);
    }

    /** init device memory **/
    ValueType *dData_cv;
    IndexType *dData_ptr;
    IndexType *dData_cid;
    int *dMembership;
    int *dMemberCount;
    ValueType *dClusters;
    int *dChanged;
    ValueType *dDst;
    ValueType *dIntermediate;

    checkCuda(cudaMalloc((void **) &dData_cv, num_nz * sizeof(ValueType)));
    checkCuda(cudaMalloc((void **) &dData_ptr, (kSrcCount + 1) * sizeof(IndexType)));
    checkCuda(cudaMalloc((void **) &dData_cid, num_nz * sizeof(IndexType)));

    checkCuda(cudaMalloc((void **) &dClusters, kClusterCount * kCoords * sizeof(ValueType)));
    checkCuda(cudaMalloc((void **) &dMembership, kSrcCount * sizeof(int)));
    checkCuda(cudaMalloc((void **) &dMemberCount, kClusterCount * sizeof(int)));
    checkCuda(cudaMalloc((void **) &dChanged, 1 * sizeof(int)));
    checkCuda(cudaMalloc((void **) &dDst, kClusterCount * kSrcCount * sizeof(ValueType)));

    if (ct == MoreParallelReduction) {
        dim3 block_size(BLOCK_DEFAULT_SIZE / 8, 8);
        dim3 grid_size((kSrcCount + block_size.x - 1) / block_size.x, kClusterCount);
        checkCuda(cudaMalloc((void **) &dIntermediate, kClusterCount * grid_size.x * kCoords * sizeof(ValueType)));
        cudaMemsetAsync(dIntermediate, 0, kClusterCount * grid_size.x * kCoords * sizeof(ValueType), stm);
    }

    checkCuda(cudaMemcpyAsync(dData_cv, Data_cv, num_nz * sizeof(ValueType), cudaMemcpyHostToDevice, stm));
    checkCuda(cudaMemcpyAsync(dData_ptr, Data_ptr, (kSrcCount + 1) * sizeof(IndexType), cudaMemcpyHostToDevice, stm));
    checkCuda(cudaMemcpyAsync(dData_cid, Data_cid, num_nz * sizeof(IndexType), cudaMemcpyHostToDevice, stm));
    checkCuda(cudaMemcpyAsync(dMembership, Membership, kSrcCount * sizeof(int), cudaMemcpyHostToDevice, stm));
    checkCuda(cudaMemcpyAsync(dMemberCount, MemberCount, kClusterCount * sizeof(int), cudaMemcpyHostToDevice, stm));
    checkCuda(cudaMemcpyAsync(dClusters, Clusters, kCoords * kClusterCount * sizeof(ValueType), cudaMemcpyHostToDevice,
                              stm));

    int itCount = 0;
    int SumNoChange = 0;
    do {
        /** update Membership **/
        checkCuda(cudaMemsetAsync(dChanged, 0, sizeof(int), stm));
        updateMembership<ValueType, IndexType, kCoords, kClusterCount, kSrcCount>(dDst, dMembership, dChanged, dData_cv,
                                                                                  dData_ptr, dData_cid, dClusters, dkt,
                                                                                  stm); checkCuda();
        cudaEventRecord(event[itCount%EventNum],stm);

        /** update MemberCount **/
        checkCuda(cudaMemsetAsync(dMemberCount,0,kClusterCount*sizeof(int),stm));
        updateMemberCount<kClusterCount,kSrcCount>(dMembership,dMemberCount,stm); checkCuda();
#if defined(DEBUG)
        checkCuda(cudaMemcpyAsync(MemberCount, dMemberCount, kClusterCount * sizeof(int), cudaMemcpyDeviceToHost,0));
        printmat(MemberCount, 1, kClusterCount, __LINE__);
#endif

        /** update clusters **/
        checkCuda(cudaMemsetAsync(dClusters,0,kClusterCount*kCoords*sizeof(ValueType),stm));
        updateClusters_cuda<ValueType,IndexType,kCoords,kClusterCount,kSrcCount>(ct,dClusters,dIntermediate,dMembership,dMemberCount,dData_cv,dData_ptr,dData_cid,stm);
#if defined(DEBUG)
        checkCuda(cudaMemcpyAsync(Clusters, dClusters, kCoords * kClusterCount * sizeof(ValueType),
                             cudaMemcpyDeviceToHost,stm));
        printmat(Clusters, kClusterCount, kCoords, __LINE__);
#endif

        checkCuda(cudaMemcpyAsync(&Changed, dChanged, sizeof(int), cudaMemcpyDeviceToHost,stm));
        SumNoChange = SumNoChange*Changed+Changed;
    } while ((SumNoChange<WaitForChange) && (itCount++ < LoopIteration));

    std::cout << "it count: " << itCount << std::endl;

    cudaFree(dData_cv);
    cudaFree(dData_ptr);
    cudaFree(dData_cid);
    cudaFree(dMembership);
    cudaFree(dChanged);
    cudaFree(dDst);
    if (ct == MoreParallelReduction) {
        cudaFree(dIntermediate);
    }

    cudaStreamDestroy(stm);
    for (size_t i = 0; i < EventNum; i++) {
        cudaEventDestroy(event[i]);
    }

    return 0;
}