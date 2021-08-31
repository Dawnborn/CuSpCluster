#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <fstream>
#include "device_launch_parameters.h"
#include "./src/utils.h"


constexpr auto WARP_SIZE = 32;
constexpr auto BLOCK_SIZE = 512;

/*TODO: move the following into the definition of class: SparseKmeans*/
constexpr int kSrcCount = 10;
constexpr int kCoords = 10;
constexpr int kClusterCount = 4;
const int loop_iteration = 40;
enum func_t {SingleAsync};

std::string path = "./../data/";

__global__ void test(void)
{
    printf("Hello CUDA!\n");
}

template<typename ValueType, typename IndexType>
int read_csr(ValueType *cv, IndexType *rp, IndexType *ci){
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

template<typename ValueType>
int get_nz(int &nz){
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

//template<typename ValueType>
//int write_csr(ValueType* ){
//    std::ofstream f1("clusters.txt");
//    std::ofstream f2("membership.txt");
//    for(int i = 0; i < kClusterCount; i++){
//        for(int j = 0; j < kCoords; j++){
//            f1<<
//        }
//    }
//}

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

/* one thread for updating one data, fixed block size */

template<typename ValueType, typename IndexType, int Coords, int SrcCount, int ClusterCount>
__global__ void KmeansMembershipUpdate_v1(
    const IndexType *drow_ptrs,
    const IndexType *dcol_idxs,
    const ValueType *dcsr_values,
    int *d_pMembership,
    const ValueType *d_pClusters,
    int *d_pChanged
){
    extern __shared__ __align__(sizeof(ValueType)) unsigned char work[];
    ValueType *sm_cluster = reinterpret_cast<ValueType *>(work); /*[ClusterCount][Coords]*/
    int *sm_blockchanged = reinterpret_cast<int*>(&sm_cluster[ClusterCount*Coords]);

    if(threadIdx.x == 0){
        *sm_blockchanged = 0;
    }

    /* load all clusters in the shared memory */
    for( int i  = threadIdx.x; i < Coords * ClusterCount; i += blockDim.x ){
        sm_cluster[i] = d_pClusters[i];
    }
    __syncthreads();

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    ValueType minDistance = 1e6;
    int clusterIndex = 0;
    int irow = tid;

    ValueType reg_Data[Coords] = {0};//
    if(irow < SrcCount){ // for latter adjustment
        int preMembership = d_pMembership[irow];
        for(IndexType i = drow_ptrs[irow]; i < drow_ptrs[irow+1]; i++){
            reg_Data[dcol_idxs[i]] = dcsr_values[i]; //FIXME: datatype
        }

        for(int itcluster = 0; itcluster < ClusterCount; itcluster++){
            ValueType curDistance = 0;
            //Compute the distance between datapoint and cluster center
            for(int idim = 0; idim < Coords; idim++){
              curDistance += std::pow(reg_Data[idim] - sm_cluster[Coords * itcluster + idim], 2);
            }
            if(curDistance < minDistance){
              minDistance = curDistance;
              clusterIndex = itcluster;
            }
        }
        bool bchanged = !(clusterIndex == preMembership); // load preMembership in register
        if(bchanged){
            d_pMembership[irow] = clusterIndex;
            atomicAdd(sm_blockchanged, 1);
        }
    }
    __syncthreads();
    if(threadIdx.x == 0){
        atomicAdd(d_pChanged, *sm_blockchanged);
    }
}

template<typename ValueType, int ClusterCount>
__global__ void KmeansClusterCenter(ValueType *d_Dst /*[Coords][ClusterCounts]*/, int *d_MemberCount){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int LocalCount;
    if(tid < ClusterCount){
        LocalCount = d_MemberCount[tid];
#pragma unroll
        for(int idim = 0; idim < kCoords; idim++){
            d_Dst[ClusterCount*idim+tid] /= LocalCount;
        }
    }
}

/*V3 减少block内同步*/ /*v5 one thread for one data*/
template<typename ValueType, typename IndexType, int BlockSize, int Coords, int SrcCount, int ClusterCount>
__global__  void KmeansClusterSum_v1(
    const IndexType *drow_ptrs,
    const IndexType *dcol_idxs,
    const ValueType *dcsr_values,
    int *d_pMembership, /*[SrcCount]*/
    ValueType *d_pDst, /*[Coords][ClusterCount]*/
    int *d_pMemberCount /*[ClusterCount]*/
){
    extern __shared__ __align__(sizeof(ValueType)) char work2[]; /*[Coords][blockDim.x] + [blockDim.x]*/ //FIXME: template and shared memory?
    int tid = threadIdx.x + blockIdx.x * BlockSize;
    ValueType *sm = reinterpret_cast<ValueType *>(work2); /*[Coords][blockDim.x], col major, better for performance*/
    int *pClusterCount = (int*)(sm + BlockSize * Coords); /*[blockDim.x]*/
    int RealCluster = d_pMembership[tid];
    if(tid < SrcCount){
        for(int itcluster = 0; itcluster < ClusterCount; itcluster++){
            bool bCurCluster = (RealCluster == itcluster);
            pClusterCount[threadIdx.x] = bCurCluster ? 1:0; //FIXME += or = ?
            // load the corresponding data in sm
#pragma unroll
            for(int idim = 0; idim < Coords; idim++){
                sm[threadIdx.x + idim * BlockSize] = (ValueType)0;
            }
            for(int i = drow_ptrs[tid]; i < drow_ptrs[tid+1]; i++){
                sm[threadIdx.x + dcol_idxs[i] * BlockSize] = dcsr_values[i];
            }

            // Reduction: out of warp
#pragma unroll
            for(int step = BlockSize/2; step >= WARP_SIZE; step /= 2){
                __syncthreads();
                if(threadIdx.x < step){
                    pClusterCount[threadIdx.x] += pClusterCount[threadIdx.x + step];
#pragma unroll
                    for(int idim = 0; idim < Coords; idim++){
                        sm[threadIdx.x + idim * BlockSize] += sm[threadIdx.x + idim * BlockSize + step];
                    }
                }
            }
            // Reduction: warp shuffle
            if(threadIdx.x < WARP_SIZE){
                int LocalClusterCount = pClusterCount[threadIdx.x];
                ValueType LocalPoints[Coords];
#pragma unroll
                for(int i = 0; i < Coords; i++){
                    LocalPoints[i] = sm[threadIdx.x + i * BlockSize];
                }
                LocalClusterCount += __shfl_down_sync(0xFFFFFFFF, LocalClusterCount, 16, 32);
                LocalClusterCount += __shfl_down_sync(0xFFFFFFFF, LocalClusterCount, 8, 32);
                LocalClusterCount += __shfl_down_sync(0xFFFFFFFF, LocalClusterCount, 4, 32);
                LocalClusterCount += __shfl_down_sync(0xFFFFFFFF, LocalClusterCount, 2, 32);
                LocalClusterCount += __shfl_down_sync(0xFFFFFFFF, LocalClusterCount, 1, 32);
#pragma unroll
                for (int i = 0; i < Coords; i++){
                    LocalPoints[i] += __shfl_down_sync(0xFFFFFFFF, LocalPoints[i], 16, 32);
                    LocalPoints[i] += __shfl_down_sync(0xFFFFFFFF, LocalPoints[i], 8, 32);
                    LocalPoints[i] += __shfl_down_sync(0xFFFFFFFF, LocalPoints[i], 4, 32);
                    LocalPoints[i] += __shfl_down_sync(0xFFFFFFFF, LocalPoints[i], 2, 32);
                    LocalPoints[i] += __shfl_down_sync(0xFFFFFFFF, LocalPoints[i], 1, 32);
                }

                if(threadIdx.x == 0){
                    atomicAdd(d_pMemberCount + itcluster, LocalClusterCount);
#pragma unroll
                    for(int idim = 0; idim < Coords; idim++){
                        atomicAdd((d_pDst + itcluster + idim * kClusterCount), LocalPoints[idim]);
                    }
                }
            }
        }
    }
}

/*single stream async*/
template<typename ValueType, typename IndexType>
void CallKmeansSingleStmAsync()
{
    /* init host memory*/
    ValueType *hcsr_values;
    IndexType *hrow_ptrs;
    IndexType *hcol_idxs;
    int *h_Membership; /*[kSrcCount]*/
    int *h_MemberCount; /*[kClusterCount]*/
    ValueType *h_Clusters;

    int num_nz; //read the data first then get the number of nonzero values
    get_nz<ValueType>(num_nz);

    /* allocate space for sparse matrix on host */
    checkCuda(cudaHostAlloc((void**)&hcsr_values, num_nz*sizeof(ValueType), cudaHostAllocDefault));
    checkCuda(cudaHostAlloc((void**)&hrow_ptrs, (kSrcCount+1)*sizeof(IndexType), cudaHostAllocDefault));
    checkCuda(cudaHostAlloc((void**)&hcol_idxs, num_nz*sizeof(IndexType), cudaHostAllocDefault));

    /* read the data from files */
    read_csr<ValueType, IndexType>(hcsr_values, hrow_ptrs, hcol_idxs);

    /* allocate and initialize the clusters */
    checkCuda(cudaHostAlloc((void**)&h_Clusters, kClusterCount*kCoords*sizeof(ValueType),cudaHostAllocDefault)); //FIXME: Initialize Cluster Center?
    std::fill(h_Clusters, h_Clusters+kClusterCount*kCoords, 0); //FIXME:fill

    /* allocate and initialize the array that store membership for all data */
    checkCuda(cudaHostAlloc((void**)&h_Membership, kSrcCount*sizeof(int), cudaHostAllocDefault)); //use fixed host memory
    std::fill(h_Membership,h_Membership+kSrcCount,0);

    /* allocate and initialize the array the store the counts of different clusters */
    checkCuda(cudaHostAlloc((void**)&h_MemberCount,kClusterCount*sizeof(int),cudaHostAllocDefault));
    std::fill(h_MemberCount, h_MemberCount+kClusterCount,0);

    /* init stream */
    cudaStream_t stm;
    cudaStreamCreate(&stm);
    const int EventNum = 10;
    cudaEvent_t event[EventNum];
    for(int i = 0; i < EventNum; i++){
        cudaEventCreate(&event[i]);
    }

    /* init device memory */
    ValueType *dcsr_values; /*[num_nz]*/
    IndexType *drow_ptrs; /*[kSrcCount+1]*/
    IndexType *dcol_idxs; /*[num_nz]*/
    ValueType *d_pClusters; /*[kClusterCount][kCoords]*/
    int *d_pChanged; /*[1]*/
    int *d_pMembership; /*[kSrcCount]*/
    int *d_pMemberCount; /*[kClusterCount]*/

    /* allocate space for sparse matrix on device */
    checkCuda(cudaMalloc((void**)&dcsr_values, num_nz*sizeof(ValueType)));
    checkCuda(cudaMalloc((void**)&drow_ptrs, (kSrcCount+1)*sizeof(IndexType)));
    checkCuda(cudaMalloc((void**)&dcol_idxs, num_nz*sizeof(IndexType)));

    checkCuda(cudaMalloc((void**)&d_pClusters, kClusterCount*kCoords*sizeof(ValueType)));
    checkCuda(cudaMalloc((void**)&d_pMembership, kSrcCount*sizeof(int)));
    checkCuda(cudaMalloc((void**)&d_pMemberCount, kClusterCount*sizeof(int)));
    checkCuda(cudaMalloc((void**)&d_pChanged, 1*sizeof(int)));

    /* copy sparse matrix from host to device */
    checkCuda(cudaMemcpyAsync(dcsr_values, hcsr_values, num_nz*sizeof(ValueType), cudaMemcpyHostToDevice, stm));
    checkCuda(cudaMemcpyAsync(drow_ptrs, hrow_ptrs, (kSrcCount+1)*sizeof(IndexType), cudaMemcpyHostToDevice, stm));
    checkCuda(cudaMemcpyAsync(dcol_idxs, hcol_idxs, num_nz*sizeof(IndexType), cudaMemcpyHostToDevice, stm));
    checkCuda(cudaMemcpyAsync(d_pMembership, h_Membership, kSrcCount*sizeof(int),cudaMemcpyHostToDevice, stm));
    checkCuda(cudaMemcpyAsync(d_pMemberCount, h_MemberCount, kClusterCount*sizeof(int),cudaMemcpyHostToDevice, stm));
    checkCuda(cudaMemcpyAsync(d_pClusters, h_Clusters, kClusterCount*kCoords*sizeof(ValueType), cudaMemcpyHostToDevice, stm));

    /* find the points */
    int itCount;
    int changed = 0;
    const int preCalcCount = 10;
    const int grid_size = (kSrcCount+BLOCK_SIZE-1)/BLOCK_SIZE;

    for(itCount = 0; itCount < preCalcCount; itCount++){
        dim3 block(BLOCK_SIZE);
        dim3 grid(grid_size);

        KmeansMembershipUpdate_v1<ValueType, IndexType, kCoords, kSrcCount, kClusterCount><<<grid,block,(kClusterCount*kCoords)*sizeof(ValueType)+sizeof(int)>>>(drow_ptrs,dcol_idxs,dcsr_values,d_pMembership,d_pClusters,d_pChanged); //TODO: membership update
        cudaError_t s = checkCuda();
        checkCuda(cudaMemcpyAsync(&changed, d_pChanged, sizeof(int), cudaMemcpyDeviceToHost, stm));

        checkCuda(cudaMemsetAsync(d_pChanged, 0, sizeof(int), stm));
        cudaEventRecord(event[itCount%10], stm);
        checkCuda(cudaMemsetAsync(d_pClusters, 0, kClusterCount*kCoords*sizeof(ValueType), stm));

        KmeansClusterSum_v1<ValueType, IndexType, BLOCK_SIZE, kCoords, kSrcCount, kClusterCount><<<grid,block,(kCoords*BLOCK_SIZE*sizeof(ValueType)+BLOCK_SIZE*sizeof(int))>>>(drow_ptrs,dcol_idxs,dcsr_values,d_pMembership,d_pClusters,d_pMemberCount); //TODO: cluster sum
        checkCuda();
        KmeansClusterCenter<ValueType, kClusterCount><<<grid,block>>>(d_pClusters,d_pMemberCount); //TODO: cluster center update
        checkCuda();

        checkCuda(cudaMemsetAsync(d_pMemberCount, 0, kClusterCount*sizeof(int),stm));
    }
    cudaEventSynchronize(event[preCalcCount-5]);

    do{
        dim3 block(BLOCK_SIZE);
        dim3 grid(grid_size);

        checkCuda(cudaMemsetAsync(d_pChanged, 0, sizeof(int), stm));

        KmeansMembershipUpdate_v1<ValueType, IndexType, kCoords, kSrcCount, kClusterCount><<<grid,block,kClusterCount*kCoords*sizeof(ValueType)+sizeof(int)>>>(drow_ptrs,dcol_idxs,dcsr_values,d_pMembership,d_pClusters,d_pChanged); //TODO: membership update
        checkCuda();

        checkCuda(cudaMemcpyAsync(&changed, d_pChanged, sizeof(int), cudaMemcpyDeviceToHost, stm));

        cudaEventRecord(event[itCount%10], stm);

        checkCuda(cudaMemsetAsync(d_pClusters, 0, kClusterCount*kCoords*sizeof(ValueType), stm));

        KmeansClusterSum_v1<ValueType, IndexType, BLOCK_SIZE, kCoords, kSrcCount, kClusterCount><<<grid,block,(kCoords*BLOCK_SIZE*sizeof(ValueType)+BLOCK_SIZE*sizeof(int))>>>(drow_ptrs,dcol_idxs,dcsr_values,d_pMembership,d_pClusters,d_pMemberCount); //TODO: cluster sum
        checkCuda();

        //FIXME: right grid and block?
        dim3 grid2((kSrcCount + BLOCK_SIZE - 1)/BLOCK_SIZE);
        KmeansClusterCenter<ValueType, kClusterCount><<<grid2,block>>>(d_pClusters,d_pMemberCount); //TODO: cluster center update
        checkCuda();

        checkCuda(cudaMemsetAsync(d_pMemberCount, 0, kClusterCount*sizeof(int), stm));

    }while((changed!=0)&&(itCount++ < loop_iteration));

    std::cout<<"it count: "<<itCount<<std::endl;

    checkCuda(cudaMemcpyAsync(h_Membership, d_pMembership, kSrcCount*sizeof(int),cudaMemcpyDeviceToHost, stm));
    checkCuda(cudaMemcpyAsync(h_MemberCount, d_pMemberCount, kClusterCount*sizeof(int),cudaMemcpyDeviceToHost, stm));
    checkCuda(cudaMemcpyAsync(h_Clusters, d_pClusters,kClusterCount*kCoords*sizeof(ValueType), cudaMemcpyDeviceToHost, stm));

    cudaFree(dcol_idxs);
    cudaFree(drow_ptrs);
    cudaFree(dcsr_values);
    cudaFree(d_pClusters);
    cudaFree(d_pChanged);
    cudaFree(d_pMembership);
    cudaFree(d_pMemberCount);
    cudaStreamDestroy(stm);
#pragma unroll
    for (int i = 0; i < EventNum; i++)
    {
        cudaEventDestroy(event[i]);
    }

    cudaFreeHost(hcol_idxs);
    cudaFreeHost(hrow_ptrs);
    cudaFreeHost(hcsr_values);
    cudaFreeHost(h_Clusters);
    cudaFreeHost(h_Membership);
    cudaFreeHost(h_MemberCount);
}

template<typename ValueType, typename IndexType>
void fit(func_t FuncIndex){
//    ValueType *h_Clusters = new ValueType[kClusterCount * kCoords];
    CallKmeansSingleStmAsync<ValueType, IndexType>();
}

int main(int, char**) {
//    test<<<1,32>>>();
    cudaDeviceSynchronize();

    CallKmeansSingleStmAsync<double,int>();

    return 0;
}
