import torch
import pytest
from unittest.mock import patch
from vptq.utils.initialize_centriods import get_centriods, weighted_kmean

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

"""
we will test the following
1. get_centriods function, to mke sure the final returned centriods are correct
2. test some intermediate transformations in the get_centriods function

"""

def test_get_centriods():
    """
    with kmeans depending on the number of centriods and the kind of data, based on the initial centriods, the final centriods can be different
    TODO: Need to check how to test this function much better way
    """
    w = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], device=DEVICE, dtype=torch.float32)
    h = torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]], device=DEVICE, dtype=torch.float32)
    k = 3
    v = 2
    expected_centriods = torch.tensor([[1, 4], [7, 8], [2, 5]], device=DEVICE, dtype=torch.float32)

    output_centriods = get_centriods(w, h, v, k)
    #assert torch.allclose(output_centriods, expected_centriods)

    

def test_weighted_kmean():
    w = torch.tensor([[1, 2], [3, 4], [15, 6], [18, 8]], device=DEVICE, dtype=torch.float32)
    h = torch.tensor([[1], [1], [1], [1]], device=DEVICE, dtype=torch.float32)
    k = 2
    expected_centriods = torch.tensor([[2, 3], [16.5, 7]], device=DEVICE, dtype=torch.float32)
    output_centriods = weighted_kmean(w, h, k)

    expected_centriods = expected_centriods[expected_centriods[:, 0].argsort()]
    output_centriods = output_centriods[output_centriods[:, 0].argsort()]
    #print(output_centriods.shape, expected_centriods.shape)
    assert torch.allclose(output_centriods, expected_centriods)


def test_get_centriods_hessian_transformation():
    h = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], device=DEVICE, dtype=torch.float32)
    expected_diag_repeated = torch.tensor([[1], [1], [5], [5], [9], [9]], device=DEVICE, dtype=torch.float32)

    def mock_weighted_kmean(w_, h_, k_):
        assert torch.allclose(h_, expected_diag_repeated)
        return True

    with patch('vptq.utils.initialize_centriods.weighted_kmean', mock_weighted_kmean):
        get_centriods(torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], device=DEVICE, dtype=torch.float32), h, 2, 3)


def test_get_centriods_weight_transformation():
    w = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], device=DEVICE, dtype=torch.float32)
    expected_w_transform = torch.tensor([[1, 4], [7, 10], [2, 5], [8, 11], [3, 6], [9, 12]], device=DEVICE, dtype=torch.float32)
    
    def mock_weighted_kmean(w_, h_, k_):
        assert torch.allclose(w_, expected_w_transform)
        return True
    
    with patch('vptq.utils.initialize_centriods.weighted_kmean', mock_weighted_kmean):
        get_centriods(w, torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], device=DEVICE, dtype=torch.float32), 2, 3)

