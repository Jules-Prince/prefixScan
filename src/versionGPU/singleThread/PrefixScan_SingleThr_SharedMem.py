from numba import cuda
import numpy as np

from PrefixScan import PrefixScan


class PrefixScan_SingleThr_SharedMem(PrefixScan):
    
    def __init__(self, array, m, n):
        super().__init__( array, m, n)
        threadsPerBlock = n
        blockPerGrid = 1
        global THREAD_BLOCK
        THREAD_BLOCK = threadsPerBlock
        d_array = cuda.to_device(array)
        self.__upSweep[blockPerGrid, threadsPerBlock](d_array, m, n)
        d_array[n-1] = 0
        self.__downSweep[blockPerGrid, threadsPerBlock](d_array, m, n)
        cuda.synchronize()

        self.array = d_array.copy_to_host()

    @cuda.jit
    def __upSweep(array, m, n) :
        shared_filter = cuda.shared.array(THREAD_BLOCK, dtype=np.int32)
        local_id = cuda.threadIdx.x
        shared_filter[cuda.threadIdx.x] = array[cuda.threadIdx.x]
        
        for d in range(0, m) :#range already does m-1
            k = local_id*(2**(d+1))
            if k<n :
                shared_filter[k+2**(d+1)-1] += shared_filter[k+2**d-1]
            cuda.syncthreads()
    
        for i in range(0, THREAD_BLOCK):
            array[i] = shared_filter[i]

    @cuda.jit
    def __downSweep(array, m, n):
        shared_filter = cuda.shared.array(THREAD_BLOCK, dtype=np.int32)
        local_id = cuda.threadIdx.x
        shared_filter[cuda.threadIdx.x] = array[cuda.threadIdx.x]
        for d in range(m-1, -1,-1 ):
            k = local_id* 2**(d+1)
            if k<n :
                a = 2**(d+1)
                b =2**d
                t = array[k+b -1]
                shared_filter[k+b -1] = shared_filter[k+a -1]
                shared_filter[k+a -1] += t
            cuda.syncthreads()
        for i in range(0, THREAD_BLOCK):
            array[i] = shared_filter[i]
