import numpy as np

def _assert(name , expected , actual , tolerance = 1e-4):
    if expected.shape != actual.shape:
        raise AssertionError( f"{ name}: missmatch "
                             f"{expected.shape}!={actual.shape}")
    

    diff =np.abs(expected-actual)
    max_diff=float(diff.max())

    if max_diff > tolerance:
        raise AssertionError(
            f"{name} difference is alot {max_diff} > {tolerance}"
        )