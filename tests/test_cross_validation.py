import pytest
import sys
import numpy as np
from unittest.mock import MagicMock
from unittest.mock import patch
sys.path.append('./src')

# from src import *
from cross_validation_correction.cross_validation import CrossValidation
from cross_validation_correction.methods import BSLMM
from bimbam import Bimbam



def mock_bslmm_predict(bslmm: BSLMM, bimbam_te: Bimbam):
    return bimbam_te.pheno

def test_cv():
    bimbam = Bimbam('./bimbam_data/bimbam_10000_full_false_major_minor.txt')
    bimbam.pheno_simulator(100)
    bimbam_tr, bimbam_te = bimbam.train_test_split()
    bimbam_gen = bimbam.fake_sample_generate(bimbam_te.n)

    with patch.object(
        BSLMM, 'predict', new=mock_bslmm_predict):
        bslmm = BSLMM()
        cv = CrossValidation(bslmm, n_folds=2)

        # fit the cv
        cv(bimbam_tr, sigmas = [1.53,0.52])
        cv.sigmas = [1.53,0.52]


        print(f'$ddd {cv.is_fitted}')
        # test predict
        re = cv.predict(bimbam_te)
        print(f'$ddd {cv.is_fitted}')
        print(f'$ddd {cv.model.bimbam.shape}')
        assert isinstance(re, np.ndarray)
        assert re.shape == bimbam_te.pheno.shape

        # Call the correct method and get the result
        result = cv.correct(bimbam_correct=bimbam_gen)

        # Assert that the result is of type int
        assert isinstance(result, np.float64), f'result is "{type(result)}" not correct'

def main():
    test_cv()

if __name__ == '__main__':
    main()