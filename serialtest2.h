#include <iostream>
#include <cuda_runtime.h>
#include <curand.h>
#include <vector>
#include <string>
#include <fstream>

#include "device_launch_parameters.h"
#include "./src/utils.h"

constexpr int GRID_SIZE = 1024;
constexpr int BLOCK_DEFAULT_SIZE = 512;
constexpr int WARP_SIZE = 32;

constexpr int kSrcCount = 30;
constexpr int kCoords = 10;
constexpr int kClusterCount = 6;
const int loop_iteration = 300;

const std::string path = "./../../data/blob/";

enum cluster_kernel_t {cpu,naive,SharedMemory,ParallelReduction,MoreParallelReduction};
enum call_func_t {};

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

template<typename ValueType, typename IndexType, int Coord, int ClusterCount, int SrcCount>
__global__ void distanceKernel1(
        ValueType* dDst /*[ClusterCount][SrcCount]*/,
        const ValueType* dData_cv,
        const IndexType* dData_ptr,
        const IndexType* dData_cid,
        const ValueType* dClusters
){
    /**
     * grid_size:
     * block_size:
     * **/
    for(unsigned int i = 0; i*gridDim.x*blockDim.x<SrcCount;i++){
        unsigned int iData = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int iCluster = blockIdx.y * blockDim.y + threadIdx.y;
        iData = iData + i * gridDim.x * blockDim.x;
        if(iData<SrcCount && iCluster<ClusterCount){
            dDst[iData + iCluster * SrcCount] = 0;
            for(unsigned int j = dData_ptr[iData]; j < dData_ptr[iData + 1]; j++){
                dDst[iData + iCluster * SrcCount] += std::pow((dClusters[iCluster * Coord + dData_cid[j]] - dData_cv[j]), 2);
            }
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
        const ValueType *dClusters
){
    cudaMemset(dChanged,0,sizeof(int));

    dim3 grid(GRID_SIZE, 16, 1), block(BLOCK_DEFAULT_SIZE, 2, 1);
    distanceKernel1<ValueType,IndexType,Coord,ClusterCount,SrcCount><<<grid,block>>>(dDst, dData_cv, dData_ptr, dData_cid, dClusters);
    checkCuda();

    checkCuda(cudaDeviceSynchronize());
//    checkCuda(cudaMemcpy(Dis,dDst,ClusterCount*SrcCount*sizeof(ValueType),cudaMemcpyDeviceToHost));

    dim3 grid2((SrcCount + BLOCK_DEFAULT_SIZE - 1) / BLOCK_DEFAULT_SIZE), block2(BLOCK_DEFAULT_SIZE);
    membershipKernel1<ValueType,ClusterCount,SrcCount><<<grid2,block2>>>(dDst, dMembership, dChanged);
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
            for (unsigned int idim = 0; idim < Coord; idim++) {
                Clusters[ClusterCount * idim + j] = (ValueType)idim;
            }
        }
    }
}

template<typename ValueType, typename IndexType, int Coord, int ClusterCount, int SrcCount>
void updateClusters_OMP(
        ValueType *Clusters, /*[Coord][ClusterCount]*/
        int *MemberCount,
        const int *Membership,
        const ValueType *Data_cv,
        const IndexType *Data_ptr,
        const IndexType *Data_cid
){} //TODO: OMP update Clusters


template<typename ValueType, int Length>
__global__ void fillKernel(ValueType *Array, ValueType val){
    for (unsigned int i = threadIdx.x; i < Length; i += blockDim.x) {
        Array[i] = val;
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
    for (unsigned int step = SrcCount / 2; step >= WARP_SIZE; step /= 2) {
        __syncthreads();
        if (tid + step < SrcCount) {
            sm[tid] += sm[tid + step];
        }
    }
    int reg_data = sm[tid];
#pragma unroll
    for (unsigned int step = WARP_SIZE / 2; step > 0; step /= 2) {
        reg_data += __shfl_xor_sync(0xffffffff, reg_data, step, WARP_SIZE);
    }

    /** write back to MemberCount **/
    if (threadIdx.x == 0) {
        if (SrcCount % 2 == 1) {
            reg_data += sm[SrcCount - 1];
        }
        dMemberCount[iCluster] = reg_data;
    }
}

template<int ClusterCount, int SrcCount>
inline void updateMemberCount(const int *dMembership, int *dMemberCount){
    dim3 grid(ClusterCount);
    dim3 block(SrcCount);
    updateMemberCountKernel<SrcCount><<<grid,block>>>(dMembership,dMemberCount);
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
     * grid_size: 1
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
        for (unsigned int i = threadIdx.x; i < Coord; i += block_size) {
            dClusters[iCluster * Coord + i] = (ValueType)i;
        }
    }
}


template<typename ValueType, typename IndexType, int Coord, int ClusterCount, int SrcCount, int BLock_SizeX>
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
        __shared__ ValueType sm[BLock_SizeX * Coord];
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


template<typename ValueType, typename IndexType, unsigned int yDim, int Coord, int ClusterCount, int SrcCount>
__global__ void updateClusterKernel_MoreParallelReduction_step1(
        ValueType *dIntermediate, /*[kClusterCount][ceil(SrcCount/BLOCK_DEFAULT_SIZE.x)][kCoord]*/
        const int *dMembership, /*[kSrcCount]*/
        const int *dMemberCount,
        const ValueType *dData_cv,
        const IndexType *dData_ptr,
        const IndexType *dData_cid
) {
    /**
     * use intermedia variable to store the reduced result from all blocks,
     * later perform reduction on them (in another new kernel)
     *
     * grid_size:(ceil(SrcCount/BLOCK_DEFAULT_SIZE.x),iCluster)
     * block_size:(BLOCK_DEFAULT_SIZE/8,8) (yDim,8)
     * **/

    unsigned int iData = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int iCluster = blockIdx.y;
    int regMemberNum = dMemberCount[iCluster];
    if (regMemberNum != 0) {
        extern __shared__ __align__ (sizeof(ValueType)) char work[];
        ValueType *sm = reinterpret_cast<ValueType *>(work); /*[block_size][Coord]*/
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
        //TODO：warp shuffle here
        for (unsigned int step = blockDim.x / 2; step > 0; step /= 2) {
            __syncthreads();
            if (iData < step) {
                for (unsigned int iDim = threadIdx.y; iDim < Coord; iDim += blockDim.y) {
                    sm[Coord * iData + iDim] += sm[Coord * (iData + step) + iDim];
                }
            }
        }
        __syncthreads();

        /** write result in the shared memory back to the intermediate **/
        if (threadIdx.y == 0) {
            for (unsigned int i = threadIdx.x; i < Coord; i += blockDim.x) {
                dIntermediate[iCluster * yDim * Coord + blockIdx.x * Coord +
                              i] = sm[i]; //FIXME: for odd number of blocks
            }
        }
        __syncthreads();
    } else {

    }
}

template<typename ValueType, typename IndexType, int Coord, int ClusterCount, int BlockNum>
__global__ void updateClusterKernel_MoreParallelReduction_step2(
        ValueType *dIntermediate, /*[kClusterCount][BlockNum][kCoord]*/
        ValueType *dClusters, /*[kClusterCount][kCoord]*/
        const int *dMemberCount
) {
    /**
     * one block for one cluster,
     *
     * grid_size: CLusterCount
     * block_size: (BLOCK_DEFAULT_SIZE/8,8)
     * **/
    unsigned int iCluster = blockIdx.x;
    ValueType *dpIntermediate = &dIntermediate[iCluster * BlockNum * Coord];
    int regMemberCount = dMemberCount[iCluster];
    if(regMemberCount!=0) {
        /** load data for reduction in the shared memory **/
        __shared__ ValueType sm[BlockNum * Coord];
        for (unsigned int i = threadIdx.x; i < BlockNum; i += blockDim.x) {
            for (unsigned int j = threadIdx.y; j < Coord; j += blockDim.y) {
                sm[Coord * i + j] = dpIntermediate[Coord * i + j];
            }
        }
        __syncthreads();

        /** perform reduction **/
        for (unsigned int step = BlockNum / 2; step > 0; step /= 2) {
            __syncthreads();
            if (threadIdx.x < step) {
                for (unsigned int iDim = threadIdx.y; iDim < Coord; iDim += blockDim.y) {
                    sm[threadIdx.x * Coord + iDim] += dpIntermediate[(threadIdx.x + step) * Coord +
                                                                     iDim]; //FIXME: singular number of data
                }
            }
        }
        __syncthreads();

        /** write back to dClusters **/
        if (threadIdx.y == 0) {
            for (unsigned int iDim = threadIdx.x; iDim < Coord; iDim += blockDim.x) {
                dClusters[iCluster * Coord + iDim] = sm[iDim] / (ValueType) regMemberCount;
            }
        }
    } else {

    }

}

template<typename ValueType, typename IndexType, int Coord, int ClusterCount, int SrcCount>
inline void updateClusters_cuda(
        cluster_kernel_t kernel_type,
        ValueType *dClusters,
        ValueType *dIntermediate,
        const int *dMembership, /*[kSrcCount]*/
        const int *dMemberCount, /*[kClusterCount]*/
        const ValueType *dData_cv,
        const IndexType *dData_ptr,
        const IndexType *dData_cid
) {
    fillKernel<ValueType, kClusterCount * kCoords><<<1, BLOCK_DEFAULT_SIZE>>>(dClusters, 0);
    if (kernel_type == naive) {
        dim3 block_size(ClusterCount, 64);
        updateClustersKernel_naive<ValueType, IndexType, Coord, ClusterCount, SrcCount><<<1, block_size>>>(
                dClusters,
                dMembership,
                dMemberCount,
                dData_cv,
                dData_ptr,
                dData_cid);
        checkCuda();
    } else if (kernel_type == SharedMemory) {
        dim3 block_size(BLOCK_DEFAULT_SIZE);
        updateClustersKernel_SharedMemory<ValueType, IndexType, Coord, ClusterCount, SrcCount><<<ClusterCount, block_size>>>(
                dClusters, dMembership, dMemberCount, dData_cv, dData_ptr, dData_cid);
        checkCuda();
    } else if (kernel_type == ParallelReduction) {
        dim3 block_size(BLOCK_DEFAULT_SIZE / 8, 8);
        dim3 grid_size((SrcCount + block_size.x - 1) / block_size.x, ClusterCount);
        updateClusterKernel_ParallelReduction1<ValueType, IndexType, Coord, ClusterCount, SrcCount,
                BLOCK_DEFAULT_SIZE / 8><<<grid_size, block_size>>>(dClusters, dMembership, dMemberCount, dData_cv,
                                                                   dData_ptr, dData_cid);
        checkCuda();
    } else if (kernel_type == MoreParallelReduction) {
        dim3 block_size(BLOCK_DEFAULT_SIZE / 8, 8);
        dim3 grid_size((SrcCount + block_size.x - 1) / block_size.x, ClusterCount);
        updateClusterKernel_MoreParallelReduction_step1<ValueType, IndexType, (SrcCount + BLOCK_DEFAULT_SIZE / 8 - 1) /
                                                                              (BLOCK_DEFAULT_SIZE /
                                                                               8), Coord, ClusterCount, SrcCount><<<grid_size, block_size,
        block_size.x * Coord * sizeof(ValueType)>>>(
                dIntermediate, dMembership, dMemberCount, dData_cv, dData_ptr, dData_cid);

        dim3 grid_size2(ClusterCount);
        dim3 block_size2(BLOCK_DEFAULT_SIZE / 8, 8);
        updateClusterKernel_MoreParallelReduction_step2<ValueType, IndexType, Coord, ClusterCount,
                (SrcCount + BLOCK_DEFAULT_SIZE / 8 - 1) / (BLOCK_DEFAULT_SIZE / 8)><<<grid_size2, block_size2>>>(
                dIntermediate,
                dClusters,
                dMemberCount);
    }
}

template<typename ValueType, typename IndexType>
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

    updateClusters_cuda<ValueType, IndexType, 10, 3, 30>(ct, dClusters, dIntermediate, dMembership, dMemberCount,
                                                         dData_cv, dData_ptr, dData_cid);

    checkCuda(cudaMemcpy(Clusters, dClusters, 30 * sizeof(ValueType), cudaMemcpyDeviceToHost));
    printmat(Clusters, 3, 10, __LINE__);

    /** test MembershipUpdate **/
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


    cudaFree(dData_cv);
    cudaFree(dData_ptr);
    cudaFree(dData_cid);
    cudaFree(dMembership);
    if(ct==MoreParallelReduction){
        cudaFree(dIntermediate);
    }

    return 0;
}

template<typename ValueType, typename IndexType>
int CallfuncSync(cluster_kernel_t ct,std::string path) {
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
    for (unsigned int i = 0; i < kClusterCount * kCoords; i++) { Clusters[i] = rand() % 10; }

    checkCuda(cudaHostAlloc((void **) &Membership, kSrcCount * sizeof(int),
                            cudaHostAllocDefault)); //use fixed host memory
    std::fill(Membership, Membership + kSrcCount, 0);

    checkCuda(cudaHostAlloc((void **) &MemberCount, kClusterCount * sizeof(int), cudaHostAllocDefault));
    std::fill(MemberCount, MemberCount + kClusterCount, 0);

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
    //kClusterCount * grid_size.x * kCoords
    if (ct == MoreParallelReduction) {
        dim3 block_size(BLOCK_DEFAULT_SIZE / 8, 8);
        dim3 grid_size((kSrcCount + block_size.x - 1) / block_size.x, kClusterCount);
        checkCuda(cudaMalloc((void **) &dIntermediate, kClusterCount * grid_size.x * kCoords * sizeof(ValueType)));
        cudaMemset(dIntermediate,0,kClusterCount * grid_size.x * kCoords * sizeof(ValueType));
    }

    checkCuda(cudaMemcpy(dData_cv, Data_cv, num_nz * sizeof(ValueType), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(dData_ptr, Data_ptr, (kSrcCount + 1) * sizeof(IndexType), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(dData_cid, Data_cid, num_nz * sizeof(IndexType), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(dMembership, Membership, kSrcCount * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(dMemberCount, MemberCount, kClusterCount * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(dClusters, Clusters, kCoords * kClusterCount * sizeof(ValueType), cudaMemcpyHostToDevice));

    int itCount = 0;
    do {
        /** update Memberships and MemberCount **/
        updateMembership<ValueType, IndexType, kCoords, kClusterCount, kSrcCount>(dDst, dMembership, dChanged, dData_cv,
                                                                                  dData_ptr, dData_cid, dClusters);
        checkCuda();
#if defined(DEBUG)
        checkCuda(cudaMemcpy(Membership, dMembership, kSrcCount * sizeof(int), cudaMemcpyDeviceToHost));
        printmat(Membership,1,kSrcCount,__LINE__);
#endif

        checkCuda(cudaMemcpy(&Changed, dChanged, sizeof(int), cudaMemcpyDeviceToHost));

        updateMemberCount<kClusterCount, kSrcCount>(dMembership, dMemberCount);
        checkCuda();
#if defined(DEBUG)
        checkCuda(cudaMemcpy(MemberCount, dMemberCount, kClusterCount * sizeof(int), cudaMemcpyDeviceToHost));
        printmat(MemberCount, 1, kClusterCount, __LINE__);
#endif

        /** update clusters **/
        if (ct == cpu) {
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
            updateClusters_cuda<ValueType, IndexType, kCoords, kClusterCount, kSrcCount>(ct, dClusters, dIntermediate,
                                                                                         dMembership, dMemberCount,
                                                                                         dData_cv, dData_ptr,
                                                                                         dData_cid);
#if defined(DEBUG)
            checkCuda(cudaMemcpy(Clusters, dClusters, kCoords * kClusterCount * sizeof(ValueType),
                                 cudaMemcpyDeviceToHost));
            printmat(Clusters, kClusterCount, kCoords, __LINE__);
#endif
        }
    } while ((Changed==1) && (itCount++ < loop_iteration));

    std::cout << "it count: " << itCount << std::endl;

    cudaFree(dData_cv);
    cudaFree(dData_ptr);
    cudaFree(dData_cid);
    cudaFree(dMembership);
    cudaFree(dChanged);
    cudaFree(dDst);
    if(ct==MoreParallelReduction){
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

template<typename ValueType, typename IndexType>
int CallfuncSingleStream(cluster_kernel_t ct, std::string path){
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
    for (unsigned int i = 0; i < kClusterCount * kCoords; i++) { Clusters[i] = rand() % 10; }

    checkCuda(cudaHostAlloc((void **) &Membership, kSrcCount * sizeof(int),
                            cudaHostAllocDefault)); //use fixed host memory
    std::fill(Membership, Membership + kSrcCount, 0);

    checkCuda(cudaHostAlloc((void **) &MemberCount, kClusterCount * sizeof(int), cudaHostAllocDefault));
    std::fill(MemberCount, MemberCount + kClusterCount, 0);

    cudaStream_t stm;
    cudaStreamCreate(&stm);
    const int EventNum = 10;
    cudaEvent_t event[EventNum];
    for(size_t i = 0; i < EventNum; i++){
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
        cudaMemsetAsync(dIntermediate,0,kClusterCount * grid_size.x * kCoords * sizeof(ValueType),stm);
    }

    checkCuda(cudaMemcpyAsync(dData_cv, Data_cv, num_nz * sizeof(ValueType), cudaMemcpyHostToDevice,stm));
    checkCuda(cudaMemcpyAsync(dData_ptr, Data_ptr, (kSrcCount + 1) * sizeof(IndexType), cudaMemcpyHostToDevice,stm));
    checkCuda(cudaMemcpyAsync(dData_cid, Data_cid, num_nz * sizeof(IndexType), cudaMemcpyHostToDevice,stm));
    checkCuda(cudaMemcpyAsync(dMembership, Membership, kSrcCount * sizeof(int), cudaMemcpyHostToDevice,stm));
    checkCuda(cudaMemcpyAsync(dMemberCount, MemberCount, kClusterCount * sizeof(int), cudaMemcpyHostToDevice,stm));
    checkCuda(cudaMemcpyAsync(dClusters, Clusters, kCoords * kClusterCount * sizeof(ValueType), cudaMemcpyHostToDevice,stm));

    int itCount = 0;
    do{
        /** update Membership and MemberCount **/
        updateMembership<ValueType,IndexType,kCoords,kClusterCount,kSrcCount>(dDst,dMembership,dChanged,dData_cv,dData_ptr,dData_cid,dClusters);

    }while(1);


}

