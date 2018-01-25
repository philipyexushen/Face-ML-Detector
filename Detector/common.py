import numpy as np
import cv2.cv2 as cv
from numba import jit

def MethodInformProvider(method):
    def _decorator(*args, **kwargs):
        import time
        t0 = time.clock()
        print(f"[Call {method.__name__}]")
        ret = method(*args, **kwargs)
        t1 = time.clock()
        print(f"[Method {method.__name__} take {t1 - t0}s to execute]")
        print(f"[Out {method.__name__}]")
        return ret
    return _decorator