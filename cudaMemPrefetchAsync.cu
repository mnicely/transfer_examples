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

/*
 * This example benchmarks copying two data arrays to and from the GPU.
 * It uses unified memory memory and chrono, cuda events, and NVTX for timing.
 */

#include "cuda_helper.h"

template<typename T>
void MemManaged<T>::run_chrono( ) {

    for ( int i = 0; i < this->loops; i++ ) {
        this->reset( N, h_a_unified.get( ), h_b_unified.get( ) );

        this->start = std::chrono::high_resolution_clock::now( );

        CUDA_RT_CALL( cudaMemPrefetchAsync( this->h_a_unified.get( ), size, this->device, this->cuda_streams[0] ) );
        CUDA_RT_CALL( cudaMemPrefetchAsync( this->h_b_unified.get( ), size, this->device, this->cuda_streams[1] ) );

        CUDA_RT_CALL( cudaLaunchKernel( reinterpret_cast<void *>( &VectorOperation<Add<T>, T> ),
                                        this->blocks_per_grid,
                                        this->threads_per_block,
                                        this->a_args,
                                        0,
                                        this->cuda_streams[0] ) );
        CUDA_RT_CALL( cudaLaunchKernel( reinterpret_cast<void *>( &VectorOperation<Sub<T>, T> ),
                                        this->blocks_per_grid,
                                        this->threads_per_block,
                                        this->b_args,
                                        0,
                                        this->cuda_streams[1] ) );

        CUDA_RT_CALL( cudaMemPrefetchAsync( this->h_a_unified.get( ), size, cudaCpuDeviceId, this->cuda_streams[0] ) );
        CUDA_RT_CALL( cudaMemPrefetchAsync( this->h_b_unified.get( ), size, cudaCpuDeviceId, this->cuda_streams[1] ) );

        CUDA_RT_CALL( cudaStreamSynchronize( this->cuda_streams[0] ) );
        CUDA_RT_CALL( cudaStreamSynchronize( this->cuda_streams[1] ) );

        this->verify( N, h_a_unified.get( ), h_b_unified.get( ) );

        this->stop = std::chrono::high_resolution_clock::now( );
        this->elapsed_chrono_ms += this->stop - this->start;
    }
    this->get_chrono_results( size );
}

template<typename T>
void MemManaged<T>::run_events( ) {

    for ( int i = 0; i < this->loops; i++ ) {
        this->reset( N, h_a_unified.get( ), h_b_unified.get( ) );

        CUDA_RT_CALL( cudaEventRecord( this->start_event ) );

        CUDA_RT_CALL( cudaMemPrefetchAsync( this->h_a_unified.get( ), size, this->device, this->cuda_streams[0] ) );
        CUDA_RT_CALL( cudaMemPrefetchAsync( this->h_b_unified.get( ), size, this->device, this->cuda_streams[1] ) );

        CUDA_RT_CALL( cudaLaunchKernel( reinterpret_cast<void *>( &VectorOperation<Add<T>, T> ),
                                        this->blocks_per_grid,
                                        this->threads_per_block,
                                        this->a_args,
                                        0,
                                        this->cuda_streams[0] ) );
        CUDA_RT_CALL( cudaLaunchKernel( reinterpret_cast<void *>( &VectorOperation<Sub<T>, T> ),
                                        this->blocks_per_grid,
                                        this->threads_per_block,
                                        this->b_args,
                                        0,
                                        this->cuda_streams[1] ) );

        CUDA_RT_CALL( cudaMemPrefetchAsync( this->h_a_unified.get( ), size, cudaCpuDeviceId, this->cuda_streams[0] ) );
        CUDA_RT_CALL( cudaMemPrefetchAsync( this->h_b_unified.get( ), size, cudaCpuDeviceId, this->cuda_streams[1] ) );

        CUDA_RT_CALL( cudaStreamSynchronize( this->cuda_streams[0] ) );
        CUDA_RT_CALL( cudaStreamSynchronize( this->cuda_streams[1] ) );

        this->verify( N, h_a_unified.get( ), h_b_unified.get( ) );

        CUDA_RT_CALL( cudaEventRecord( this->stop_event ) );
        CUDA_RT_CALL( cudaEventSynchronize( this->stop_event ) );
        CUDA_RT_CALL( cudaEventElapsedTime( &this->elapsed_events_ms, this->start_event, this->stop_event ) );

        this->average_events_ms += this->elapsed_events_ms;
    }
    this->get_events_results( size );
}

template<typename T>
void MemManaged<T>::run_nvtx( ) {

    for ( int i = 0; i < this->loops; i++ ) {

        PUSH_RANGE( "Reset", 0 )
        this->reset( N, h_a_unified.get( ), h_b_unified.get( ) );
        POP_RANGE( )

        PUSH_RANGE( "Process_Loop", 1 )

        PUSH_RANGE( "H2D_A", 2 )
        CUDA_RT_CALL( cudaMemPrefetchAsync( this->h_a_unified.get( ), size, this->device, this->cuda_streams[0] ) );
        POP_RANGE( )

        PUSH_RANGE( "H2D_B", 3 )
        CUDA_RT_CALL( cudaMemPrefetchAsync( this->h_b_unified.get( ), size, this->device, this->cuda_streams[1] ) );
        POP_RANGE( )

        PUSH_RANGE( "Kernel_A", 4 )
        CUDA_RT_CALL( cudaLaunchKernel( reinterpret_cast<void *>( &VectorOperation<Add<T>, T> ),
                                        this->blocks_per_grid,
                                        this->threads_per_block,
                                        this->a_args,
                                        0,
                                        this->cuda_streams[0] ) );
        POP_RANGE( )

        PUSH_RANGE( "Kernel_B", 5 )
        CUDA_RT_CALL( cudaLaunchKernel( reinterpret_cast<void *>( &VectorOperation<Sub<T>, T> ),
                                        this->blocks_per_grid,
                                        this->threads_per_block,
                                        this->b_args,
                                        0,
                                        this->cuda_streams[1] ) );
        POP_RANGE( )

        PUSH_RANGE( "D2H_A", 6 )
        CUDA_RT_CALL( cudaMemPrefetchAsync( this->h_a_unified.get( ), size, cudaCpuDeviceId, this->cuda_streams[0] ) );
        POP_RANGE( )

        PUSH_RANGE( "D2H_B", 7 )
        CUDA_RT_CALL( cudaMemPrefetchAsync( this->h_b_unified.get( ), size, cudaCpuDeviceId, this->cuda_streams[1] ) );
        POP_RANGE( )

        CUDA_RT_CALL( cudaStreamSynchronize( this->cuda_streams[0] ) );
        CUDA_RT_CALL( cudaStreamSynchronize( this->cuda_streams[1] ) );

        PUSH_RANGE( "Verify", 8 )
        this->verify( N, h_a_unified.get( ), h_b_unified.get( ) );
        POP_RANGE( )

        POP_RANGE( )
    }
}

/* Main */
int main( int argc, char **argv ) {

    using dtype = float;

    int N = MemCpy<dtype>::init_size;
    if ( argc > 1 ) {
        N = std::atoi( argv[1] );
    }

    MemManaged<dtype> MemManaged( N );

    double gigabytes { MemManaged.size * 1e-9 };

    printf( "Running with = %lu B (%0.2f GB)\n\n", MemManaged.size, gigabytes );

    // Chrono
    MemManaged.run_chrono( );

    // Events
    MemManaged.run_events( );

    // NVTX
    MemManaged.run_nvtx( );

    return ( EXIT_SUCCESS );
}