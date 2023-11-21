import pytest
import sys
import numpy as np
from unittest.mock import MagicMock
from unittest.mock import patch
sys.path.append('./src')

# from src import *
from cross_validation_simulation import cross_validation_simulation 
from cross_validation_correction.methods import BSLMM
from bimbam import Bimbam


def mock_bslmm_predict(bslmm: BSLMM, bimbam_te: Bimbam):
    return bimbam_te.pheno

def test_cross_validation_simulation():
    bimbam_path = './bimbam_data/bimbam_10000_full_false_major_minor.txt'
    save_path = './simulation_output/test'
    n_folds = 3

    with patch.object(
        BSLMM, 'predict', new=mock_bslmm_predict):
        cross_validation_simulation(bimbam_path, save_path, n_folds=n_folds,simulation_times=2)
    
if __name__ == '__main__':
    test_cross_validation_simulation()