# Getting started

The purpose of these examples is to help demonstrates various transfer and timing techniques.

## Transfer techniques
1. Manual cudaMemcpy
2. Manual cudaMemcpyAsync
3. Unified memory
4. Unified memory with cudaMemPrefetchAsync

## Timing techniques
1. Chrono
2. CUDA events
3. NVTX markers

## Usage
Each test does the follow:
1. Transfers two chunks of data to the GPU.
2. Run simple kernel on each chunk (mutually exclusive)
3. Transfer both chunks back to CPU.
4. Verify results

Throughput based on entire workflow.

Default transfer size is 1GB.
```bash
./cudaMemcpyAsync
Running with = 1073741824 B (1.07 GB)

Chrono: 194.463501 ms @ 5.521560 GB/s
Events: 195.145355 ms @ 5.502267 GB/s
```
or add number of values to transfer
```bash
./cudaMemcpyAsync 1000000000
Running with = 4000000000 B (4.00 GB)

Chrono: 725.302307 ms @ 5.514942 GB/s
Events: 726.420593 ms @ 5.506452 GB/s
```

### Using NVTX markers
We must use Nsight Systems to see NVTX.
Open `*.qdrep` file with Nsight Systems GUI.

```bash
nsys profile -s none -t cuda,nvtx --stats=true ./cudaMemcpyAsync
WARNING: Backtraces will not be collected because sampling is disabled.
Collecting data...
Running with = 1073741824 B (1.07 GB)

Chrono: 195.130112 ms @ 5.502697 GB/s
Events: 195.151413 ms @ 5.502096 GB/s
Processing events...
Capturing symbol files...
Saving temporary "/tmp/nsys-report-8b68-3d6e-a890-0843.qdstrm" file to disk...
Creating final output files...

Processing [==============================================================100%]
Saved report file to "/tmp/nsys-report-8b68-3d6e-a890-0843.qdrep"
Exporting 1532 events: [==================================================100%]

Exported successfully to
/tmp/nsys-report-8b68-3d6e-a890-0843.sqlite


CUDA API Statistics:

 Time(%)  Total Time (ns)  Num Calls     Average       Minimum      Maximum              Name          
 -------  ---------------  ---------  -------------  -----------  -----------  ------------------------
    92.8    7,935,281,846         30  264,509,394.9  162,979,208  367,487,697  cudaStreamSynchronize   
     3.8      328,299,372          2  164,149,686.0  164,072,205  164,227,167  cudaHostAlloc           
     1.9      158,486,349          2   79,243,174.5   79,163,730   79,322,619  cudaFreeHost            
     1.5      130,534,008          2   65,267,004.0      701,703  129,832,305  cudaMalloc              
     0.0        1,454,937          2      727,468.5      691,604      763,333  cudaFree                
     0.0          401,077         60        6,684.6        2,374       26,921  cudaMemcpyAsync         
     0.0          373,116         30       12,437.2        4,637       31,135  cudaLaunchKernel        
     0.0           83,993         10        8,399.3        5,491       10,170  cudaEventRecord         
     0.0           18,491          5        3,698.2        3,500        3,931  cudaEventSynchronize    
     0.0            7,857          2        3,928.5          873        6,984  cudaStreamCreate        
     0.0            7,470          2        3,735.0        1,390        6,080  cudaStreamDestroy       
     0.0            5,474          2        2,737.0          397        5,077  cudaEventCreateWithFlags
     0.0            2,602          2        1,301.0          417        2,185  cudaEventDestroy        



CUDA Kernel Statistics:

 Time(%)  Total Time (ns)  Instances    Average     Minimum    Maximum                                    Name                                  
 -------  ---------------  ---------  -----------  ---------  ---------  -----------------------------------------------------------------------
    50.4       67,362,653         15  4,490,843.5  4,469,005  4,518,252  void VectorOperation<Add<float>, float>(int, float, float*, Add<float>)
    49.6       66,399,104         15  4,426,606.9  4,391,821  4,462,860  void VectorOperation<Sub<float>, float>(int, float, float*, Sub<float>)



CUDA Memory Operation Statistics (by time):

 Time(%)  Total Time (ns)  Operations     Average       Minimum      Maximum        Operation     
 -------  ---------------  ----------  -------------  -----------  -----------  ------------------
    50.8    5,314,309,114          30  177,143,637.1  162,980,699  191,320,991  [CUDA memcpy DtoH]
    49.2    5,142,402,501          30  171,413,416.7  168,996,768  173,680,811  [CUDA memcpy HtoD]



CUDA Memory Operation Statistics (by size in KiB):

     Total       Operations     Average        Minimum        Maximum         Operation     
 --------------  ----------  -------------  -------------  -------------  ------------------
 31,457,280.000          30  1,048,576.000  1,048,576.000  1,048,576.000  [CUDA memcpy DtoH]
 31,457,280.000          30  1,048,576.000  1,048,576.000  1,048,576.000  [CUDA memcpy HtoD]



NVTX Push-Pop Range Statistics:

 Time(%)  Total Time (ns)  Instances     Average       Minimum      Maximum       Range    
 -------  ---------------  ---------  -------------  -----------  -----------  ------------
    67.6    3,889,587,729          5  777,917,545.8  777,031,465  778,565,183  Process_Loop
    21.7    1,247,130,464          5  249,426,092.8  247,611,878  250,019,880  Verify      
    10.7      616,467,941          5  123,293,588.2  120,541,180  132,044,582  Reset       
     0.0          101,082          5       20,216.4       16,613       22,645  H2D_A       
     0.0           93,874          5       18,774.8       18,020       21,060  Kernel_A    
     0.0           29,242          5        5,848.4        5,317        6,468  Kernel_B    
     0.0           23,568          5        4,713.6        4,374        4,970  D2H_A       
     0.0           18,801          5        3,760.2        3,283        4,147  H2D_B       
     0.0           14,849          5        2,969.8        2,680        3,182  D2H_B       

Report file moved to "/home/belt/workStuff/git_examples/transfer_examples/report4.qdrep"
Report file moved to "/home/belt/workStuff/git_examples/transfer_examples/report4.sqlite"
```
