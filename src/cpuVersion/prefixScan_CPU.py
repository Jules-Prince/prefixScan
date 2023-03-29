
class prefixScan_CPU(object):
    
    def __init__(self, array, m, n):
        self.array, self.m, self.n = array, m, n
        self.__upSweep()
        self.__downSweep()

    def __upSweep(self):
        for d in range(0, self.m) :#range already does m-1
            for k in range(0, self.n-1, 2**(d+1)) :
                self.array[k+2**(d+1)-1] += self.array[k+2**d-1]

    def __downSweep(self):
        self.array[self.n-1] = 0
        for d in range(self.m-1, -1, -1) :
            for k in range(0, self.n, 2**(d+1)) :
                tmpVal = self.array[k+2**d-1]
                self.array[k+2**d-1] = self.array[k+2**(d+1)-1]
                self.array[k+2**(d+1)-1] += tmpVal