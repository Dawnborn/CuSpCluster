#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <fstream>
#include "device_launch_parameters.h"
#include "./src/utils.h"

constexpr int kSrcCount = 30;
constexpr int kCoords = 10;
constexpr int kClusterCount = 6;
const int loop_iteration = 50;
constexpr int grid_size = 1024;
constexpr int block_size = 512;

const std::string path = "./../data/blob/";

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

template<typename ValueType, typename IndexType, int Coord, int ClusterCount, int SrcCount>
__global__ void distanceKernel1(
        ValueType* c /*[ClusterCount][SrcCount]*/,
        const ValueType* a_cv,
        const IndexType* a_ptrs,
        const IndexType* a_cid,
        const ValueType* b
){
    for(int i = 0; i*gridDim.x*blockDim.x<SrcCount;i++){ //FIXME: SrcCount-1 ?
        int iData = blockIdx.x * blockDim.x + threadIdx.x;
        int iCluster = blockIdx.y * blockDim.y + threadIdx.y;
        iData = iData + i * gridDim.x * blockDim.x;
        if(iData<SrcCount && iCluster<ClusterCount){
            c[iData+iCluster*SrcCount] = 0;
            for(int j = a_ptrs[iData]; j < a_ptrs[iData+1]; j++){
                c[iData+iCluster*SrcCount] += std::pow((b[iCluster*Coord+a_cid[j]] - a_cv[j]),2);
            }
        }
    }
}

template<typename ValueType, int ClusterCount, int SrcCount>
__global__ void membershipKernel1(
        ValueType *Dst, /*[ClusterCount][SrcCount]*/
        int *Membership,
        bool *dChanged
){
    int iData = threadIdx.x + blockDim.x * blockIdx.x;
    bool flag_changed = false;
    bool regdChanged = *dChanged;
    if(iData < SrcCount){
        int min_cluster = 0;
        int pre_cluster = Membership[iData];
        ValueType min_dist = Dst[SrcCount*min_cluster + iData];
        for(int iCluster = 1; iCluster < ClusterCount; iCluster++){
            if(Dst[SrcCount*iCluster + iData] < min_dist){
                min_cluster = iCluster;
                min_dist = Dst[SrcCount*iCluster + iData];
            }
        }
        if(min_cluster!=pre_cluster) {
            Membership[iData] = min_cluster;
            flag_changed  = true;
        }
        if(flag_changed&&(regdChanged==false)){
            *dChanged = true;
        }
    }
}

template<typename ValueType, typename IndexType, int Coord, int ClusterCount, int SrcCount>
inline void updateMembership(
        ValueType *dDis, /*[ClusterCount][SrcCount]*/
        int *dMembership, /*[SrcCount]*/
        bool *dChanged,
        const ValueType *dData_cv,
        const IndexType *dData_ptr,
        const IndexType *dData_cid,
        const ValueType *dClusters
){
    dim3 grid(grid_size,16,1), block(block_size,2,1);
    distanceKernel1<ValueType,IndexType,Coord,ClusterCount,SrcCount><<<grid,block>>>(dDis,dData_cv,dData_ptr,dData_cid,dClusters);
    checkCuda();

    checkCuda(cudaDeviceSynchronize());
//    checkCuda(cudaMemcpy(Dis,dDis,ClusterCount*SrcCount*sizeof(ValueType),cudaMemcpyDeviceToHost));

    dim3 grid2((SrcCount+block_size-1)/block_size), block2(block_size);
    membershipKernel1<ValueType,ClusterCount,SrcCount><<<grid2,block2>>>(dDis,dMembership,dChanged);
    checkCuda();
    checkCuda(cudaDeviceSynchronize());
}

template<typename ValueType, typename IndexType, int Coord, int ClusterCount, int SrcCount, int Block_size>
__global__ void Sum_kernel(
        ValueType *dClusters,
        const int *dMembership,
        const int *dMembershipCount,
        const ValueType *dData_cv,
        const IndexType *dData_ptr,
        const IndexType *dData_cid
){
    __shared__ ValueType tmp[Coord*Block_size]; /*[Coord][ClusterCount]*/

    int tid = threadIdx.x + blockDim.x*blockIdx.x;
    if(tid<SrcCount){
        int cluster = dMembership[tid];
        for(int i = dData_ptr[tid]; i < dData_ptr[tid+1]; i++){
            continue;
        }
    }
}

template<typename ValueType, typename IndexType, int Coord, int ClusterCount, int SrcCount>
void updateClusters_cpu(
        ValueType *Clusters, /*[Coord][ClusterCount]*/
        int *MemberCount,
        const int *Membership,
        const ValueType *Data_cv,
        const IndexType *Data_ptr,
        const IndexType *Data_cid
){
    std::fill(Clusters,Clusters+Coord*ClusterCount,0);
    std::fill(MemberCount,MemberCount+ClusterCount,0);
    for(int i = 0; i < SrcCount; i++){
        MemberCount[Membership[i]] += 1;
        for(int j = Data_ptr[i]; j < Data_ptr[i+1]; j++){
            Clusters[ClusterCount*Data_cid[j]+Membership[i]] += Data_cv[j];
        }
    }

    for(int j = 0; j < ClusterCount; j++){
        if(MemberCount[j]!=0) {
            for (int idim = 0; idim < Coord; idim++) {
                Clusters[ClusterCount * idim + j] /= MemberCount[j];
            }
        }
    }
}

template<typename ValueType, typename IndexType, int Coord, int ClusterCount>
inline void updateClusters_gpu(
        ValueType *dClusters,
        const int *dMembership, /*[kSrcCount]*/
        const int *dMembershipCount, /*[kClusterCount]*/
        const ValueType *dData_cv,
        const IndexType *dData_ptr,
        const IndexType *dData_cid
){
    
}

template<typename ValueType, typename IndexType>
void CallfuncCPU(){
    ValueType *Data_cv;
    IndexType *Data_ptr;
    IndexType *Data_cid;
    int *Membership;
    int *MemberCount;
    ValueType *Clusters;
    bool Changed;

    int num_nz;
    get_nz<ValueType>(num_nz);

    checkCuda(cudaHostAlloc((void**)&Data_cv, num_nz*sizeof(ValueType), cudaHostAllocDefault));
    checkCuda(cudaHostAlloc((void**)&Data_ptr, (kSrcCount+1)*sizeof(IndexType), cudaHostAllocDefault));
    checkCuda(cudaHostAlloc((void**)&Data_cid, num_nz*sizeof(IndexType), cudaHostAllocDefault));

    read_csr<ValueType, IndexType>(Data_cv, Data_ptr, Data_cid);

    checkCuda(cudaHostAlloc((void**)&Clusters, kClusterCount*kCoords*sizeof(ValueType),cudaHostAllocDefault)); //FIXME: Initialize Cluster Center?
//    std::fill(Clusters, Clusters+kClusterCount*kCoords, 1); //FIXME:fill
    for(int i = 0; i < kClusterCount*kCoords;i++){
        Clusters[i] = i%kClusterCount;
    }

    checkCuda(cudaHostAlloc((void**)&Membership, kSrcCount*sizeof(int), cudaHostAllocDefault)); //use fixed host memory
    std::fill(Membership,Membership+kSrcCount,0);

    checkCuda(cudaHostAlloc((void**)&MemberCount,kClusterCount*sizeof(int),cudaHostAllocDefault));
    std::fill(MemberCount, MemberCount+kClusterCount,0);

    ValueType *dData_cv;
    IndexType *dData_ptr;
    IndexType *dData_cid;
    int *dMembership;
//    int *dMemberCount;
    ValueType *dClusters;
    bool *dChanged;

    checkCuda(cudaMalloc((void**)&dData_cv, num_nz*sizeof(ValueType)));
    checkCuda(cudaMalloc((void**)&dData_ptr, (kSrcCount+1)*sizeof(IndexType)));
    checkCuda(cudaMalloc((void**)&dData_cid, num_nz*sizeof(IndexType)));

    checkCuda(cudaMalloc((void**)&dClusters, kClusterCount*kCoords*sizeof(ValueType)));
    checkCuda(cudaMalloc((void**)&dMembership, kSrcCount*sizeof(int)));
//    checkCuda(cudaMalloc((void**)&dMemberCount, kClusterCount*sizeof(int)));
    checkCuda(cudaMalloc((void**)&dChanged, 1*sizeof(bool)));

    checkCuda(cudaMemcpy(dData_cv,Data_cv,num_nz*sizeof(ValueType),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(dData_ptr,Data_ptr,(kSrcCount+1)*sizeof(IndexType),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(dData_cid,Data_cid,num_nz*sizeof(IndexType),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(dMembership,Membership,kSrcCount*sizeof(int),cudaMemcpyHostToDevice));
//    checkCuda(cudaMemcpy(dMemberCount,MemberCount,kClusterCount*sizeof(int),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(dClusters,Clusters,kCoords*kClusterCount*sizeof(ValueType),cudaMemcpyHostToDevice));

    ValueType *dDst;
    checkCuda(cudaMalloc((void**)&dDst, kClusterCount*kSrcCount*sizeof(ValueType)));
    ValueType *Dst = new ValueType[kClusterCount*kSrcCount]{0};

    int itCount = 0;
//    int changed = 0;
//    const int preCalcCount=5;

    do{
        updateMembership<ValueType,IndexType,kCoords,kClusterCount,kSrcCount>(dDst,dMembership,dChanged,dData_cv,dData_ptr,dData_cid,dClusters);
        checkCuda(cudaMemcpy(&Changed,dChanged,sizeof(bool),cudaMemcpyDeviceToHost));
        checkCuda(cudaMemcpy(Membership,dMembership,kSrcCount*sizeof(int),cudaMemcpyDeviceToHost));
        checkCuda(cudaMemcpy(Dst,dDst,kClusterCount*kSrcCount*sizeof(ValueType),cudaMemcpyDeviceToHost));

        updateClusters_cpu<ValueType,IndexType,kCoords,kClusterCount,kSrcCount>(Clusters,MemberCount,Membership,Data_cv,Data_ptr,Data_cid);
        checkCuda(cudaMemcpy(dClusters,Clusters,kCoords*kClusterCount*sizeof(ValueType),cudaMemcpyHostToDevice));

    }while((Changed==true)&&(itCount++<loop_iteration));

    std::cout<<"it count: "<<itCount<<std::endl;

    cudaFree(dData_cv);
    cudaFree(dData_ptr);
    cudaFree(dData_cid);
    cudaFree(dMembership);
    cudaFree(dChanged);
    cudaFree(dDst);

    cudaFreeHost(Data_cv);
    cudaFreeHost(Data_ptr);
    cudaFree(Data_cid);
    cudaFreeHost(Membership);
    cudaFreeHost(MemberCount);
    cudaFreeHost(Clusters);
}

int main(){
    CallfuncCPU<double,int>();
}
