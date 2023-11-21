import sys
sys.path.append('./src')    
from bimbam import Bimbam
from cross_validation_correction.methods import BSLMM
import numpy as np
import pytest

def test_BSLMM():
    bimbam = Bimbam('./bimbam_data/bimbam_10000_full_false_major_minor.txt')
    bimbam.pheno_simulator(100)
    bimbam_tr, bimbam_te = bimbam.train_test_split()
    bimbam_gen = bimbam.fake_sample_generate(bimbam_te.n)

    bslmm = BSLMM(sigmas=[0.5, 0.5])
    bslmm.fit(bimbam_tr, bimbam_tr.pheno)
    try:
        y=bslmm.predict(bimbam_te)
    except FileNotFoundError as f:
        pass
    h_te = bslmm.generate_h_te(bimbam_te)
    assert h_te.shape == (bimbam_te.n, bimbam_tr.n),\
        f'Wrong shape for h_te {h_te.shape}, should be ({bimbam_te.n}, {bimbam_tr.n})'

    h_gen = bslmm.generate_h_te(bimbam_gen)
    assert h_gen.shape == (bimbam_te.n, bimbam_tr.n),\
        f'Wrong shape for h_te {h_gen.shape}, should be ({bimbam_te.n}, {bimbam_tr.n})'
