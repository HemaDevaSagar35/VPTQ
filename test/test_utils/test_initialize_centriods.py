import torch
import pytest
from VPTQ.utils.initialize_centriods import get_centriods

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

"""
we will test the following
1. get_centriods function, to mke sure the final returned centriods are correct
2. test some intermediate transformations in the get_centriods function

"""

def test_get_centriods():
    return True


def test_get_centriods_hessian_transformation():
    h = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], device=DEVICE)
    expected_diag_repeated = torch.tensor([[1], [1], [5], [5], [9], [9]], device=DEVICE)

    def mock_weighted_kmean(w_, h_, k_):
        assert torch.allclose(h_, expected_diag_repeated)
        return True

    with patch('VPTQ.utils.initialize_centriods.weighted_kmean', mock_weighted_kmean):
        get_centriods(torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], device=DEVICE), h, 2, 3)


def test_get_centriods_weight_transformation():
    w = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], device=DEVICE)
    expected_w_transform = torch.tensor([[1, 4], [7, 10], [2, 5], [8, 11], [3, 6], [9, 12]], device=DEVICE)
    
    def mock_weighted_kmean(w_, h_, k_):
        assert torch.allclose(w_, expected_w_transform)
        return True
    
    with patch('VPTQ.utils.initialize_centriods.weighted_kmean', mock_weighted_kmean):
        get_centriods(w, torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], device=DEVICE), 2, 3)

