from numba import cuda

from PrefixScan import PrefixScan

class PrefixScan_SingleThr(PrefixScan):
    
    def __init__(self, array, m, n):
        super().__init__( array, m, n)
        threadsPerBlock = n
        blockPerGrid = 1

        d_array = cuda.to_device(array)
        self.__upSweep[blockPerGrid, threadsPerBlock](d_array, m, n)
        d_array[n-1] = 0
        self.__downSweep[blockPerGrid, threadsPerBlock](d_array, m, n)
        cuda.synchronize()

        self.array = d_array.copy_to_host()

    @cuda.jit
    def __upSweep(array, m, n) :
        local_id = cuda.threadIdx.x
        for d in range(0, m) :#range already does m-1a
            k = local_id*(2**(d+1))
            if k<n :
                array[k+2**(d+1)-1] += array[k+2**d-1]
            cuda.syncthreads()

    @cuda.jit
    def __downSweep(array, m, n):
        local_id = cuda.threadIdx.x
        
        for d in range( m-1, -1, -1) :
            k = local_id*(2**(d+1))
            if k<n :
                tmpVal =  array[k+2**d-1]
                array[k+2**d-1] =  array[k+2**(d+1)-1]
                array[k+2**(d+1)-1] += tmpVal
            cuda.syncthreads()