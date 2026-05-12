import numpy as np
import pytest

from MultivariateBLF.MultivariateBLF import FunctionalData

class TestCoordsToMatrixIfNot_ArrayBranch:
    def test_1d_coords_become_2d_and_data_passthrough(self):
        coords = np.array([0.0, 1.0, 2.0])            # 1D -> should become (3,1)
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # 2D

        out_coords, out_data = FunctionalData._coords_to_matrix_if_not(coords, data)

        assert out_coords.shape == (3, 1)
        np.testing.assert_array_equal(out_coords[:, 0], np.array([0.0, 1.0, 2.0]))
        np.testing.assert_array_equal(out_data, data)

    def test_data_must_be_2d_raises(self):
        coords = np.array([[0.0], [1.0], [2.0]])
        bad_data = np.array([1.0, 2.0, 3.0])  # 1D
        with pytest.raises(ValueError, match="data must be 2D array"):
            FunctionalData._coords_to_matrix_if_not(coords, bad_data)

    def test_coords_gt_2d_raises(self):
        coords = np.zeros((3, 1, 1))  # ndim = 3
        data = np.zeros((3, 2))
        with pytest.raises(ValueError, match="coords must be 1D or 2D array"):
            FunctionalData._coords_to_matrix_if_not(coords, data)

    def test_first_dim_mismatch_raises(self):
        coords = np.array([[0.0], [1.0], [2.0]])  # shape (3,1)
        data = np.zeros((4, 2))                   # first dim mismatch
        with pytest.raises(ValueError, match="first dimension of coords must match"):
            FunctionalData._coords_to_matrix_if_not(coords, data)

    def test_inf_converted_to_nan_and_warns(self):
        coords = np.array([[0.0], [1.0]])
        data = np.array([[1.0, np.inf], [np.nan, -np.inf]])

        with pytest.warns(UserWarning, match="contains Inf values"):
            out_coords, out_data = FunctionalData._coords_to_matrix_if_not(coords, data)

        # coords unchanged (already 2D)
        np.testing.assert_array_equal(out_coords, coords)

        # All non-finite -> NaN (NaN stays NaN, ±Inf become NaN)
        assert np.isnan(out_data[0, 1])
        assert np.isnan(out_data[1, 0])
        assert np.isnan(out_data[1, 1])
        assert out_data[0, 0] == 1.0


class TestCoordsToMatrixIfNot_ListBranch:
    def test_lists_must_have_same_length(self):
        coords = [np.array([[0.0]]), np.array([[1.0]])]
        data = [np.array([1.0])]
        with pytest.raises(ValueError, match="same length"):
            FunctionalData._coords_to_matrix_if_not(coords, data)

    def test_if_coords_is_list_data_must_be_list(self):
        coords = [np.array([[0.0]])]
        data = np.array([[1.0]])  # not a list
        with pytest.raises(ValueError, match="data must also be a list"):
            FunctionalData._coords_to_matrix_if_not(coords, data)

    def test_if_data_is_list_coords_must_be_list(self):
        coords = np.array([[0.0]])  # not a list
        data = [np.array([1.0])]
        with pytest.raises(ValueError, match="data must also be an array"):
            FunctionalData._coords_to_matrix_if_not(coords, data)

    def test_each_data_must_be_1d(self):
        coords = [np.array([[0.0], [1.0]])]
        data = [np.array([[1.0], [2.0]])]  # 2D
        with pytest.raises(ValueError, match="each data array must be 1D"):
            FunctionalData._coords_to_matrix_if_not(coords, data)

    def test_coords_ndim_gt2_raises(self):
        coords = [np.zeros((2, 1, 1))]  # ndim=3
        data = [np.array([1.0, 2.0])]
        with pytest.raises(ValueError, match="1D or 2D"):
            FunctionalData._coords_to_matrix_if_not(coords, data)

    def test_coords_second_dim_inconsistent_raises(self):
        coords = [np.array([[0.0, 0.0], [1.0, 0.0]]), np.array([[0.0], [1.0]])]
        data = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        with pytest.raises(ValueError, match="same second dimension"):
            FunctionalData._coords_to_matrix_if_not(coords, data)

    def test_coords_and_data_lengths_must_match_per_sample(self):
        coords = [np.array([[0.0, 0.0], [1.0, 0.0]])]
        data = [np.array([1.0])]  # mismatch
        with pytest.raises(ValueError, match="first dimension of each coords array must match"):
            FunctionalData._coords_to_matrix_if_not(coords, data)

    def test_list_branch_rejects_nan_or_inf_in_data(self):
        coords = [np.array([[0.0, 0.0], [1.0, 0.0]])]
        data = [np.array([1.0, np.inf])]
        with pytest.raises(ValueError, match="must not contain NaN or Inf"):
            FunctionalData._coords_to_matrix_if_not(coords, data)

    def test_lexicographic_merge_and_nan_fill(self):
        # Two samples with overlapping and unique coords in 2D; data are 1D
        c0 = np.array([[0.0, 0.0], [1.0, 0.0]])
        d0 = np.array([10.0, 20.0])
        c1 = np.array([[0.0, 0.0], [0.0, 1.0]])
        d1 = np.array([30.0, 40.0])

        out_coords, out_data = FunctionalData._coords_to_matrix_if_not([c0, c1], [d0, d1])

        # Unique coords sorted lexicographically: (0,0), (0,1), (1,0)
        expected_coords = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        np.testing.assert_array_equal(out_coords, expected_coords)

        # Data aligned to coord rows; NaN where a sample lacks that coord
        expected_data = np.array([
            [10.0, 30.0],   # (0,0)
            [np.nan, 40.0], # (0,1)
            [20.0, np.nan], # (1,0)
        ])
        # Use assert_allclose with equal_nan=True to compare NaNs
        np.testing.assert_allclose(out_data, expected_data, equal_nan=True)

    def test_current_behavior_list_with_1d_coords_errors(self):
        """
        Document the present bug: when coords in the list are 1D, the function
        converts them to 2D only inside the validation loop, but later re-iterates
        the original 1D arrays and tries tuple(row), where row is a scalar -> TypeError.
        """
        coords = [np.array([0.0, 1.0])]
        data = [np.array([1.0, 2.0])]

        with pytest.raises(TypeError):
            FunctionalData._coords_to_matrix_if_not(coords, data)


