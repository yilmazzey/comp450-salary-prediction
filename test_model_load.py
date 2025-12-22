import numpy
import numpy.random._pcg64

class PatchedPCG64(numpy.random._pcg64.PCG64):
    def __setstate__(self, state):
        return

def __total_override_ctor(name):
    return PatchedPCG64()

# Patch NumPy BEFORE importing joblib
import numpy.random._pickle
numpy.random._pickle.__bit_generator_ctor = __total_override_ctor
numpy.random._pickle.bit_generator_ctor = __total_override_ctor

import joblib

print('Numpy version:', numpy.__version__)
try:
    model = joblib.load('data/processed/best_model.joblib')
    print('Model loaded successfully')
except Exception as e:
    import traceback
    traceback.print_exc()
