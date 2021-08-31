#include <cuda_runtime.h>

template<typename ValueType, typename IndexType, int Coords, int SrcCount, int ClusterCount>
__global__ void KmeansMembershipUpdate_v1(
        const IndexType *drow_ptrs,
        const IndexType *dcol_idxs,
        const ValueType *dcsr_values,
        int *d_pMembership,
        const ValueType *d_pClusters,
        int *d_pChanged
){}

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