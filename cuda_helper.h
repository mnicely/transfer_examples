/*
 * Copyright 1993-2020 NVIDIA Corporation.  All rights reserved.
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

#include <cassert>
#include <chrono>
#include <cstdio>
#include <memory>
#include <vector>

// *************** FOR ERROR CHECKING *******************
#ifndef CUDA_RT_CALL
#define CUDA_RT_CALL( call )                                                                                           \
    {                                                                                                                  \
        auto status = static_cast<cudaError_t>( call );                                                                \
        if ( status != cudaSuccess )                                                                                   \
            std::fprintf( stderr,                                                                                      \
                          "ERROR: CUDA RT call \"%s\" in line %d of file %s failed "                                   \
                          "with "                                                                                      \
                          "%s (%d).\n",                                                                                \
                          #call,                                                                                       \
                          __LINE__,                                                                                    \
                          __FILE__,                                                                                    \
                          cudaGetErrorString( status ),                                                                \
                          status );                                                                                    \
    }
#endif  // CUDA_RT_CALL
// *************** FOR ERROR CHECKING *******************

// ***************** FOR NVTX MARKERS *******************
#ifdef USE_NVTX
#include "nvtx3/nvToolsExt.h"

const uint32_t colors[]   = { 0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff00ff, 0xff00ffff, 0xffff0000, 0xffffffff };
const int      num_colors = sizeof( colors ) / sizeof( uint32_t );

#define PUSH_RANGE( name, cid )                                                                                        \
    {                                                                                                                  \
        int color_id                      = cid;                                                                       \
        color_id                          = color_id % num_colors;                                                     \
        nvtxEventAttributes_t eventAttrib = { 0 };                                                                     \
        eventAttrib.version               = NVTX_VERSION;                                                              \
        eventAttrib.size                  = NVTX_EVENT_ATTRIB_STRUCT_SIZE;                                             \
        eventAttrib.colorType             = NVTX_COLOR_ARGB;                                                           \
        eventAttrib.color                 = colors[color_id];                                                          \
        eventAttrib.messageType           = NVTX_MESSAGE_TYPE_ASCII;                                                   \
        eventAttrib.message.ascii         = name;                                                                      \
        nvtxRangePushEx( &eventAttrib );                                                                               \
    }
#define POP_RANGE( ) nvtxRangePop( );
#else
#define PUSH_RANGE( name, cid )
#define POP_RANGE( )
#endif
// ***************** FOR NVTX MARKERS *******************

// ****************** CLASS Add **********************
template<class O>
class Add {
  public:
    __host__ __device__ O operator( )( const O &a, const O &b ) const {
        return a + b;
    }
};
// ****************** CLASS Add **********************

// ****************** CLASS Sub **********************
template<class O>
class Sub {
  public:
    __host__ __device__ O operator( )( const O &a, const O &b ) const {
        return a - b;
    }
};
// ****************** CLASS Sub **********************

template<class O, typename T>
__global__ void VectorOperation( const int n, const T a, T *__restrict__ x, O op ) {
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x ) {
        // printf("%d: %f %f %f\n", i, a, x[i], op( a, x[i] ));
        x[i] = op( a, x[i] );
    }
}

// ****************** CLASS MemCpy **********************
template<typename T>
class MemCpy {
  public:
    MemCpy( );
    MemCpy( const size_t &N );
    ~MemCpy( ) noexcept;

    virtual void run_chrono( ) = 0;
    virtual void run_events( ) = 0;
    virtual void run_nvtx( )   = 0;

    const static size_t init_size { 1024 * 1024 * 256 };

  protected:
    // Host variables
    const static size_t loops { 5 };
    std::vector<T>      key_A {};
    std::vector<T>      key_B {};

    // Timing variables: Chrono
    std::chrono::high_resolution_clock::time_point start {};
    std::chrono::high_resolution_clock::time_point stop {};
    std::chrono::duration<float, std::milli>       elapsed_chrono_ms {};
    float                                          average_chrono_ms {};

    // Timing variables: Events
    cudaEvent_t start_event { nullptr };
    cudaEvent_t stop_event { nullptr };
    float       elapsed_events_ms {};
    float       average_events_ms {};

    double throughput {};

    // CUDA streams
    const static int num_streams { 2 };
    cudaStream_t     cuda_streams[num_streams];

    // Helper functions
    T *PagedAllocate( const size_t &N );

    struct PagedMemoryDeleter {
        void operator( )( T *ptr );
    };

    using UniquePagedPtr = std::unique_ptr<T, PagedMemoryDeleter>;

    // Device variables
    UniquePagedPtr d_a {};
    UniquePagedPtr d_b {};

    T a { 5.0f };
    T b { 9.0f };

    int device {};
    int threads_per_block { 512 };
    int blocks_per_grid {};

    // Host functions
    void get_chrono_results( const size_t &size );
    void get_events_results( const size_t &size );
    void reset( const size_t &N, T *a, T *b );
    void verify( const size_t &N, T *a, T *b );

    // Kernel Arguments
    Add<T> *add_op {};
    Sub<T> *sub_op {};
    void *  a_args[4];
    void *  b_args[4];

  private:
    int sm_count {};

    void fill( const size_t &N, const T &x, T *input );
};

template<typename T>
MemCpy<T>::MemCpy( ) : add_op( new Add<T>( ) ), sub_op( new Sub<T>( ) ) {

    CUDA_RT_CALL( cudaGetDevice( &device ) );
    CUDA_RT_CALL( cudaDeviceGetAttribute( &sm_count, cudaDevAttrMultiProcessorCount, device ) );

    CUDA_RT_CALL( cudaEventCreate( &start_event, cudaEventBlockingSync ) );
    CUDA_RT_CALL( cudaEventCreate( &stop_event, cudaEventBlockingSync ) );

    for ( int i = 0; i < num_streams; i++ ) {
        CUDA_RT_CALL( cudaStreamCreate( &cuda_streams[i] ) );
    }

    blocks_per_grid = sm_count * 32;
}

template<typename T>
MemCpy<T>::MemCpy( const size_t &N ) :
    add_op( new Add<T>( ) ),
    sub_op( new Sub<T>( ) ),
    d_a { PagedAllocate( N ) },
    d_b { PagedAllocate( N ) } {

    CUDA_RT_CALL( cudaGetDevice( &device ) );
    CUDA_RT_CALL( cudaDeviceGetAttribute( &sm_count, cudaDevAttrMultiProcessorCount, device ) );

    CUDA_RT_CALL( cudaEventCreate( &start_event, cudaEventBlockingSync ) );
    CUDA_RT_CALL( cudaEventCreate( &stop_event, cudaEventBlockingSync ) );

    for ( int i = 0; i < num_streams; i++ ) {
        CUDA_RT_CALL( cudaStreamCreate( &cuda_streams[i] ) );
    }

    blocks_per_grid = sm_count * 32;
}

template<typename T>
MemCpy<T>::~MemCpy( ) noexcept {

    CUDA_RT_CALL( cudaEventDestroy( start_event ) );
    CUDA_RT_CALL( cudaEventDestroy( stop_event ) );

    for ( int i = 0; i < num_streams; i++ ) {
        CUDA_RT_CALL( cudaStreamDestroy( cuda_streams[i] ) );
    }

    delete[] add_op;
    delete[] sub_op;
}

template<typename T>
T *MemCpy<T>::PagedAllocate( const size_t &N ) {
    T *    ptr { nullptr };
    size_t bytes { N * sizeof( T ) };
    CUDA_RT_CALL( cudaMalloc( reinterpret_cast<void **>( &ptr ), bytes ) );
    return ( ptr );
}

template<typename T>
void MemCpy<T>::PagedMemoryDeleter::operator( )( T *ptr ) {
    if ( ptr ) {
        CUDA_RT_CALL( cudaFree( ptr ) );
    }
};

template<typename T>
void MemCpy<T>::get_chrono_results( const size_t &size ) {
    average_chrono_ms = elapsed_chrono_ms.count( ) / ( loops * 4 );
    throughput        = ( size * 1e-9 ) / ( average_chrono_ms * 0.001 );
    std::printf( "Chrono: %0.6f ms @ %0.6f GB/s\n", average_chrono_ms, throughput );
}

template<typename T>
void MemCpy<T>::get_events_results( const size_t &size ) {
    average_events_ms /= ( loops * 4 );
    throughput = ( size * 1e-9 ) / ( average_events_ms * 0.001 );
    std::printf( "Events: %0.6f ms @ %0.6f GB/s\n", average_events_ms, throughput );
}

template<typename T>
void MemCpy<T>::fill( const size_t &N, const T &x, T *input ) {
    for ( int i = 0; i < N; i++ ) {
        input[i] = x;
    }
}

template<typename T>
void MemCpy<T>::reset( const size_t &N, T *x, T *y ) {
    this->fill( N, a, x );
    this->fill( N, b, y );
}

template<typename T>
void MemCpy<T>::verify( const size_t &N, T *x, T *y ) {
    for ( int i = 0; i < N; i++ ) {
        assert( x[i] == add_op->operator( )( b, a ) );
        assert( y[i] == sub_op->operator( )( a, b ) );
    }
}
// ****************** CLASS MemCpy **********************

// **************** CLASS MemCpyPaged *******************
template<typename T>
class MemCpyPaged : public MemCpy<T> {
  public:
    MemCpyPaged( ) noexcept = delete;
    MemCpyPaged( const size_t &N );
    ~MemCpyPaged( ) noexcept;

    void run_chrono( );
    void run_events( );
    void run_nvtx( );

    size_t size {};

  private:
    // Host variables
    std::vector<T> h_a_paged {};
    std::vector<T> h_b_paged {};

    size_t N {};
};

template<typename T>
MemCpyPaged<T>::MemCpyPaged( const size_t &N ) :
    MemCpy<T>( N ),
    N { N },
    size { N * sizeof( T ) },
    h_a_paged( N ),
    h_b_paged( N ) {

    this->a_args[0] = reinterpret_cast<void *>( &this->N );
    this->a_args[1] = &this->b;
    this->a_args[2] = &this->d_a;
    this->a_args[3] = &this->add_op;

    this->b_args[0] = reinterpret_cast<void *>( &this->N );
    this->b_args[1] = &this->a;
    this->b_args[2] = &this->d_b;
    this->b_args[3] = &this->sub_op;
}

template<typename T>
MemCpyPaged<T>::~MemCpyPaged( ) noexcept {}
// **************** CLASS MemCpyPaged *******************

// **************** CLASS MemCpyPinned *******************
template<typename T>
class MemCpyPinned : public MemCpy<T> {
  public:
    MemCpyPinned( ) noexcept = delete;
    MemCpyPinned( const size_t &N );
    ~MemCpyPinned( ) noexcept;

    void run_chrono( );
    void run_events( );
    void run_nvtx( );

    size_t size {};

  private:
    // Helper functions
    T *PinnedAllocate( const size_t &N );

    struct PinnedMemoryDeleter {
        void operator( )( T *ptr );
    };

    using UniquePinnedPtr = std::unique_ptr<T, PinnedMemoryDeleter>;

    // Host variables
    UniquePinnedPtr h_a_pinned {};
    UniquePinnedPtr h_b_pinned {};

    size_t N {};
};

template<typename T>
MemCpyPinned<T>::MemCpyPinned( const size_t &N ) :
    MemCpy<T>( N ),
    N { N },
    size { N * sizeof( T ) },
    h_a_pinned { PinnedAllocate( N ) },
    h_b_pinned { PinnedAllocate( N ) } {

    this->a_args[0] = reinterpret_cast<void *>( &this->N );
    this->a_args[1] = &this->b;
    this->a_args[2] = &this->d_a;
    this->a_args[3] = &this->add_op;

    this->b_args[0] = reinterpret_cast<void *>( &this->N );
    this->b_args[1] = &this->a;
    this->b_args[2] = &this->d_b;
    this->b_args[3] = &this->sub_op;
}

template<typename T>
MemCpyPinned<T>::~MemCpyPinned( ) noexcept {}

template<typename T>
T *MemCpyPinned<T>::PinnedAllocate( const size_t &N ) {
    T *    ptr { nullptr };
    size_t bytes { N * sizeof( T ) };
    CUDA_RT_CALL( cudaHostAlloc( reinterpret_cast<void **>( &ptr ), bytes, cudaHostAllocDefault ) );
    return ( ptr );
}

template<typename T>
void MemCpyPinned<T>::PinnedMemoryDeleter::operator( )( T *ptr ) {
    if ( ptr ) {
        CUDA_RT_CALL( cudaFreeHost( ptr ) );
    }
};
// **************** CLASS MemCpyPinned *******************

// **************** CLASS MemManaged *****************
template<typename T>
class MemManaged : public MemCpy<T> {
  public:
    MemManaged( ) noexcept = delete;
    MemManaged( const size_t &N );
    ~MemManaged( ) noexcept;

    void run_chrono( );
    void run_events( );
    void run_nvtx( );

    size_t size {};

  private:
    // Helper functions
    template<typename U>
    U *UnifiedAllocate( const size_t &N );

    template<typename U>
    struct UnifiedMemoryDeleter {
        void operator( )( U *ptr );
    };

    template<typename U>
    using UniqueUnifiedPtr = std::unique_ptr<U, UnifiedMemoryDeleter<U>>;

    // Host variables
    UniqueUnifiedPtr<T> h_a_unified {};
    UniqueUnifiedPtr<T> h_b_unified {};

    size_t N {};
};

template<typename T>
MemManaged<T>::MemManaged( const size_t &N ) :
    MemCpy<T>( ),
    N { N },
    size { N * sizeof( T ) },
    h_a_unified { UnifiedAllocate<T>( N ) },
    h_b_unified { UnifiedAllocate<T>( N ) } {

    this->a_args[0] = reinterpret_cast<void *>( &this->N );
    this->a_args[1] = &this->b;
    this->a_args[2] = &this->h_a_unified;
    this->a_args[3] = &this->add_op;

    this->b_args[0] = reinterpret_cast<void *>( &this->N );
    this->b_args[1] = &this->a;
    this->b_args[2] = &this->h_b_unified;
    this->b_args[3] = &this->sub_op;
}

template<typename T>
MemManaged<T>::~MemManaged( ) noexcept {}

template<typename T>
template<typename U>
U *MemManaged<T>::UnifiedAllocate( const size_t &N ) {
    U *    ptr { nullptr };
    size_t bytes { N * sizeof( U ) };
    CUDA_RT_CALL( cudaMallocManaged( reinterpret_cast<void **>( &ptr ), bytes ) );
    return ( ptr );
}

template<typename T>
template<typename U>
void MemManaged<T>::UnifiedMemoryDeleter<U>::operator( )( U *ptr ) {
    if ( ptr ) {
        CUDA_RT_CALL( cudaFree( ptr ) );
    }
};
// **************** CLASS MemManaged *****************
