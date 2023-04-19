from pytest import fixture
import numpy as np
from median_polish import median_polish
from median_polish.main import med_abs_dev

TEST_DATA_FOLDER = "median_polish/test/data.npz"

@fixture
def simple_table():
    return np.array([[25.3, 25.3, 18.2, 18.3, 16.3],
                     [32.1, 29.0, 18.8, 24.3, 19.0],
                     [38.8, 31.0, 19.3, 15.7, 16.8],
                     [25.4, 21.1, 20.3, 24.0, 17.5]])

def test_simple(simple_table):
    result_1 = median_polish(simple_table, 1)
    correct_1 = np.array([[-0.4, 1.15, 0.35, 0, 1.05],
                          [0.4, -1.15, -5.05, 0, -2.25],
                          [12.1, 5.85, 0.45, -3.6, 0.55],
                          [-3.1, -5.85, -0.35, 2.9, -0.55]])
    effects_1 = [np.array([7.4, 5.85, -0.45, 0, -3.05]), np.array([-1.9, 4.1, -0.9, 0.9])]
    assert(np.isclose(result_1['ave'], 20.2))
    assert(np.allclose(result_1['column'], effects_1[1]))
    assert(np.allclose(result_1['row'], effects_1[0]))
    assert(np.allclose(result_1['r'], correct_1))
    result_2 = median_polish(simple_table, 2)
    correct_2 = np.array([[-1.15, 0.4, 0.05, -0.75, 0.7],
                          [1.15, -0.4, -3.85, 0.75, -1.1],
                          [11.15, 4.9, -0.05, -4.55, 0],
                          [-2.95, -5.7, 0.25, 3.05, 0]])
    effects_2 = [np.array([7.8, 6.25, -0.5, 0.4, -3.05]), np.array([-1.55, 2.95, -0.35, 0.35])]
    assert(np.isclose(result_2['ave'], 20.2))
    assert(np.allclose(result_2['column'], effects_2[1]))
    assert(np.allclose(result_2['row'], effects_2[0]))
    assert(np.allclose(result_2['r'], correct_2))
    assert(np.allclose(median_polish(simple_table, 3)['ave'], 20.6))

def test_median_absolute_deviation(simple_table):
    result = med_abs_dev(simple_table)
    assert result == 3.75
