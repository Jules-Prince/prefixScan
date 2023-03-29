from numba import cuda
import numba as nb
import numpy as np

from PrefixScan import PrefixScan

class prefixScan_CPU(PrefixScan):
    
    def __init__(self, array, m, n):
        super().__init__( array, m, n)
        threadsPerBlock = n
        blockPerGrid = 1

        d_array = cuda.to_device(array)
        self.__upSweep_SingleThread[blockPerGrid, threadsPerBlock](d_array, m, n)
        self.__downSweep_SingleThread[blockPerGrid, threadsPerBlock](d_array, m, n)
        cuda.synchronize()

        self.array = d_array.copy_to_host()

    @cuda.jit
    def __upSweep_SingleThread(array, m, n) :
        local_id = cuda.threadIdx.x
        for d in range(0, m) :#range already does m-1a
            k = local_id*(2**(d+1))
            if k<n :
                array[k+2**(d+1)-1] += array[k+2**d-1]

    @cuda.jit
    def __downSweep_SingleThread(array, m, n):
        local_id = cuda.threadIdx.x
        array[n-1] = 0
        for d in range( m-1, -1, -1) :
            k = local_id*(2**(d+1))
            if k<n :
                tmpVal =  array[k+2**d-1]
                array[k+2**d-1] =  array[k+2**(d+1)-1]
                array[k+2**(d+1)-1] += tmpVal