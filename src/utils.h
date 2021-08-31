//
// Created by hjp on 2021/8/12.
//
#include <iostream>
#include <assert.h>

inline cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
    if (result != cudaSuccess) {
        fprintf(stderr, "\n CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
#endif
return result;
}

inline cudaError_t checkCuda(){
#if defined(DEBUG) || defined(_DEBUG)
    cudaError_t result;
    result = cudaGetLastError();
    if (result != cudaSuccess) {
        fprintf(stderr, "\n CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
#endif
}

template<typename ValueType>
void pin(ValueType mark){
    std::ofstream flog;
    flog.open("./log.csv", std::ofstream::app);
    flog<<"line: "<<mark<<" passed."<<std::endl;
}

template<typename ValueType>
inline void pin(int line, ValueType *p, const size_t len){
    std::ofstream flog;
    flog.open("./log.csv", std::ofstream::app);
    flog<<"vector at "<<line<<std::endl<<"[";
    for(size_t i = 0; i < len; i++){
        flog<<p[i]<<",";
    }
    flog<<"]"<<std::endl;
}

template<typename ValueType>
inline void pin(ValueType *p, const size_t num_rows, const size_t num_cols, int line){
    std::ofstream flog;
    flog.open("./log.csv", std::ofstream::app);
    flog<<"vector at "<<line<<std::endl<<"["<<std::endl;
    for(size_t i = 0; i < num_rows; i++){
        for(size_t j = 0; j < num_cols; j++){
            flog<<p[num_cols*i+j]<<",";
        }
        flog<<std::endl;
    }
    flog<<"]"<<std::endl;
}

template<typename ValueType>
inline void writemat(ValueType *A, const size_t num_rows, const size_t num_cols, int mark){
    std::ofstream mat_csv;
    mat_csv.open("./mat.csv", std::ofstream::app);
    mat_csv<<"mat: "<<mark<<" num_rows: "<<num_rows<<" num_cols: "<<num_cols<<std::endl;
    for(size_t i = 0; i < num_rows; i++){
        for(size_t j = 0; j < num_cols; j++ ){
            mat_csv<<A[num_cols*i+j]<<",";
        }
        mat_csv<<"]"<<std::endl<<std::endl;
    }
}
