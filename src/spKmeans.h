#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include "device_launch_parameters.h"
#include "utils.h"
#include "kernels.h"

enum func_t {SingleAsync};

template<typename ValueType>
inline int get_nz(path){
    std::ifstream ff;
    ff.open((path+"tcsr_ptr.txt"), std::ifstream::in);
    int count = 0;
    ValueType tmp;
    while(ff>>tmp){
        count++;
    }
    ff.close();
    this->num_nz = tmp;
    return tmp;
}

template<typename ValueType, typename IndexType>
inline int read_csr(ValueType *cv, IndexType *rp, IndexType *ci, std::string path){
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
    }else{
        std::cout<<"fail to open csv_values!"<<std::endl;
        return -1;
    }
    if(f2){
        for(count = 0; f2>>idx; count++){
            ci[count] = idx;
        }
        f2.close();//pin(ci,1,num_nz,__LINE__);
    }else{
        std::cout<<"fail to open col_idx!"<<std::endl;
        return -1;
    }

    if(f3){
        for(count = 0; f3>>idx ;count++){
            rp[count] = idx;
        }
        f3.close();
    }else{
        std::cout<<"fail to open row_ptrs!"<<std::endl;
        return -1;
    }
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

template<typename ValueType, typename IndexType>
class spKmeans {
private:
    int kCoords;
    int kClusterCount;
    int kSrcCount;
    int num_nz;

    int loop_iterations;
    float threshold;
    std::string path;

    const IndexType *hrow_ptrs;
    const IndexType *hcol_idxs;
    const ValueType *hcsr_values;

    int *kMembership;
    ValueType *kClusters;
    int *kMemberCount;

    constexpr int BLOCK_SIZE = 512;

    int CallKmeansSingleStmAsync();
    int


public:
    spKmeans(int pkCoords, int pkClusterCount, int ploop_iterations):
        kCoords(pkCoords),
        kClusterCount(pkClusterCount),
        loop_iterations(ploop_iterations),
        {};
    int fit(func_t id);
};



template<typename ValueType, typename IndexType>
int spKmeans<ValueType,IndexType>::fit(func_t id){
    if(id == SingleAsync) {
        this->CallKmeansSingleStmAsync();
    }
    return 0;
}


template<typename ValueType, typename IndexType>
int spKmeans<ValueType, IndexType>::CallKmeansSingleStmAsync(){

    this->num_nz = get_nz<ValueType>(this->path);

    /*allocate space for sparse matrix on host*/
    checkCuda(cudaHostAlloc((void**)&hcsr_values, num_nz*sizeof(ValueType), cudaHostAllocDefault));
    checkCuda(cudaHostAlloc((void**)&hrow_ptrs, (kSrcCount+1)*sizeof(IndexType), cudaHostAllocDefault));
    checkCuda(cudaHostAlloc((void**)&hcol_idxs, num_nz*sizeof(IndexType), cudaHostAllocDefault));
    read_csr(hcsr_values,hrow_ptrs,hcol_idxs,path);

    checkCuda(cudaHostAlloc((void**)&h_Clusters, kClusterCount*kCoords*sizeof(ValueType),cudaHostAllocDefault)); //FIXME: Initialize Cluster Center?
    std::fill(h_Clusters, h_Clusters+kClusterCount*kCoords, 0);
    checkCuda(cudaHostAlloc((void**)&h_Membership, kSrcCount*sizeof(int), cudaHostAllocDefault)); //use fixed host memory
    std::fill(h_Membership,h_Membership+kSrcCount,0);
    checkCuda(cudaHostAlloc((void**)&h_MemberCount,kClusterCount*sizeof(int),cudaHostAllocDefault));
    std::fill(h_MemberCount, h_MemberCount+kClusterCount,0);

    /*init stream*/
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

        KmeansMembershipUpdate_v1<ValueType, IndexType, kCoords><<<grid,block,(kClusterCount*kCoords)*sizeof(ValueType)+sizeof(int)>>>(kSrcCount,kClusterCount,drow_ptrs,dcol_idxs,dcsr_values,d_pMembership,d_pClusters,d_pChanged); //TODO: membership update
        checkCuda();
        checkCuda(cudaMemcpyAsync(&changed, d_pChanged, sizeof(int), cudaMemcpyDeviceToHost, stm));
        checkCuda(cudaMemsetAsync(d_pChanged, 0, sizeof(int), stm));
        cudaEventRecord(event[itCount%10], stm);
        checkCuda(cudaMemsetAsync(d_pClusters, 0, kClusterCount*kCoords*sizeof(ValueType), stm));

        KmeansClusterSum_v1<ValueType, IndexType, kCoords, BLOCK_SIZE><<<grid,block,(kCoords*BLOCK_SIZE*sizeof(ValueType)+BLOCK_SIZE*sizeof(int))>>>(kSrcCount,kClusterCount,drow_ptrs,dcol_idxs,dcsr_values,d_pMembership,d_pClusters,d_pMemberCount); //TODO: cluster sum
        checkCuda();
        KmeansClusterCenter<ValueType><<<grid,block>>>(d_pClusters,d_pMemberCount); //TODO: cluster center update
        checkCuda();
        checkCuda(cudaMemsetAsync(d_pMemberCount, 0, kClusterCount*sizeof(int),stm));
    }
    cudaEventSynchronize(event[preCalcCount-5]);

    do{
        dim3 block(BLOCK_SIZE);
        dim3 grid(grid_size);

        cudaMemsetAsync(d_pChanged, 0, sizeof(int), stm);

        KmeansMembershipUpdate_v1<ValueType, IndexType, kCoords><<<grid,block,kClusterCount*kCoords*sizeof(ValueType)+sizeof(int)>>>(kSrcCount,kClusterCount,drow_ptrs,dcol_idxs,dcsr_values,d_pMembership,d_pClusters,d_pChanged); //TODO: membership update
        checkCuda();

        cudaMemcpyAsync(&changed, d_pChanged, sizeof(int), cudaMemcpyDeviceToHost, stm);

        cudaEventRecord(event[itCount%10], stm);

        cudaMemsetAsync(d_pClusters, 0, kClusterCount*kCoords*sizeof(ValueType), stm);

        KmeansClusterSum_v1<ValueType, IndexType, kCoords, BLOCK_SIZE><<<grid,block,(kCoords*BLOCK_SIZE*sizeof(ValueType)+BLOCK_SIZE*sizeof(int))>>>(kSrcCount,kClusterCount,drow_ptrs,dcol_idxs,dcsr_values,d_pMembership,d_pClusters,d_pMemberCount); //TODO: cluster sum
        checkCuda();

        //FIXME: right grid and block?
        dim3 grid2((kSrcCount + BLOCK_SIZE - 1)/BLOCK_SIZE);
        KmeansClusterCenter<ValueType><<<grid2,block>>>(d_pClusters,d_pMemberCount); //TODO: cluster center update
        checkCuda();

        cudaMemsetAsync(d_pMemberCount, 0, kClusterCount*sizeof(int), stm);

    }while((changed!=0)&&(itCount++ < loop_iteration));

}

