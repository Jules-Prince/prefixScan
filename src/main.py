from PIL import Image
from numba import cuda
import numba as nb
import numpy as np
import time
from cpuVersion.prefixScan_CPU import prefixScan_CPU


def main():
    m = 4
    n = 2**m
    #array = np.array([2, 3, 4, 6])
    array = np.random.randint(0, 10, n, dtype=np.int32)
    print("before :", array)
    start_time = time.time()
    prefixScan = prefixScan_CPU(array, m, n)
    end_time = time.time()
    print("after  :", prefixScan.array)
    print("--- %s seconds ---" % (end_time - start_time))



if __name__ == '__main__':
    main()