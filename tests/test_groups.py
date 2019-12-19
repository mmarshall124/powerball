from src import groups
from numpy import array
from numpy import array_equal
from numpy import unique


def test_group_submatrix():
    # Testing if the array generated by group_submatrix
    # matches the expected values
    data = array([[20, 50, 10, 20, 0], [30, 30, 30, 10, 0],
                  [10, 15, 20, 45, 10], [5, 5, 5, 0, 85],
                  [15, 0, 15, 10, 0], [20, 0, 20, 15, 5]])
    groupArray = array(["C", "C", "C", "H", "H", "H"], dtype=str)
    groupName = "C"
    submatrix = array([[20, 50, 10, 20, 0], [30, 30, 30, 10, 0],
                       [10, 15, 20, 45, 10]])

    assert array_equal(groups.group_submatrix(data, groupArray,
                                              groupName), submatrix), "Generated submatrix doesn't match expected values."


def test_group_submatrices():
    # Testing if the array of group submatrices matches the expected values
    data = array([[20, 50, 10, 20, 0], [30, 30, 30, 10, 0],
                  [10, 15, 20, 45, 10], [5, 5, 5, 0, 85],
                  [15, 0, 15, 10, 0], [20, 0, 20, 15, 5]])
    groupArray = array(["C", "C", "C", "H", "H", "H"], dtype=str)

    submatrices = array([[[20, 50, 10, 20, 0], [30, 30, 30, 10, 0],
                          [10, 15, 20, 45, 10]],
                         [[5, 5, 5, 0, 85], [15, 0, 15, 10, 0],
                          [20, 0, 20, 15, 5]]])
    testUnique = ["C", "H"]

    outputSubmatrices, outputUnique = groups.group_submatrices(data,
                                                               groupArray,
                                                               3)

    assert array_equal(outputSubmatrices, submatrices), "Generated submatrices don't match expected values."
    assert array_equal(outputUnique, testUnique), "Filtered Unique lists don't match"
