import math
import time
import numpy as np
import sys

from versionGPU.singleThread.PrefixScan_SingleThr_SharedMem import PrefixScan_SingleThr_SharedMem
from versionGPU.singleThread.PrefixScan_SingleThr import PrefixScan_SingleThr
from versionCPU.PrefixScan_CPU import PrefixScan_CPU

THREAD_BLOCk = 16

def computeTimeElapsed(start, end) :
    timeDif = end-start
    print("elapsed in ", end="")
    if timeDif > 1 :
        print(timeDif, "s")
        return
    elif timeDif >= 10**-3:
        print(timeDif*10**3, "ms")
        return
    elif timeDif >= 10**-6:
        print(timeDif*10**6, "us")
        return
           

def run_CPU(array, m, n):
    start_time = time.time()
    arr =  PrefixScan_CPU(array, m, n).array
    end_time = time.time()
    print("CPU ", end="")
    computeTimeElapsed(start_time, end_time)
    return arr

def run_GPU_SingleThr(array, m, n):
    start_time = time.time()
    arr =  PrefixScan_SingleThr(array, m, n).array
    end_time = time.time()
    print("GPU ", end="")
    computeTimeElapsed(start_time, end_time)
    return arr

def run_GPU_SingleThr_SharedMem(array, m, n):
    start_time = time.time()

    arr =  PrefixScan_SingleThr_SharedMem(array, m, n).array
    end_time = time.time()
    print("GPU ", end="")
    computeTimeElapsed(start_time, end_time)
    return arr

def main(sizeArray):
    m = int(math.log2(sizeArray))
    n = 2**m
    array = np.array([2, 3, 4, 6])
    array = np.random.randint(0, 10, sizeArray, np.int32)
    scanCPUArr = run_CPU(np.copy(array), m, n)
    scanGPUArr = run_GPU_SingleThr(np.copy(array), m, n)
    scanGPUSharedArr = run_GPU_SingleThr_SharedMem(np.copy(array), m, n)

    print("Array            :", array)
    print("CPUArray         :", scanCPUArr)
    print("GPUArray         :", scanGPUArr)
    print("GPU Shared Array :", scanGPUSharedArr)

    isEquals = False
    if np.array_equal(scanCPUArr, scanGPUArr):
        if np.array_equal(scanGPUArr, scanGPUSharedArr) :
            isEquals = True
    print("Equals ? -->", np.array_equal(scanCPUArr, scanGPUArr) and np.array_equal(scanCPUArr, scanGPUSharedArr))

if __name__ == '__main__':
    main(int(sys.argv[1]))