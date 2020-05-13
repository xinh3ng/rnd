"""
"""
from copy import deepcopy
import numpy as np
from typing import List


def matrix_str(matrix: List[List[int]]):
    return np.array(matrix)


class SolutionOne:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.

        In this approach, I record all the cells that should be set to 0. In a separate for loop, I will zero-set
        the corresponding cells
        """
        n_rows, n_cols = len(matrix), len(matrix[0])
        zeros = set()
        for row in range(n_rows):
            for col in range(n_cols):
                if matrix[row][col] != 0:
                    continue
                set_one = set([(x, col) for x in range(n_rows)])
                zeros = zeros.union(set_one)

                set_two = set([(row, y) for y in range(n_cols)])
                zeros = zeros.union(set_two)

        for x in zeros:
            matrix[x[0]][x[1]] = 0


class SolutionTwo:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.

        Similar to SolutionOne, but I don't use the complicated set functions

        """
        n_rows, n_cols = len(matrix), len(matrix[0])
        zero_rows, zero_cols = set(), set()

        for row in range(n_rows):
            for col in range(n_cols):
                if matrix[row][col] != 0:
                    continue

                zero_rows.add(row)
                zero_cols.add(col)

        for row in zero_rows:
            for col in range(n_cols):
                if matrix[row][col] == 0:
                    continue
                matrix[row][col] = 0

        for row in range(n_rows):
            for col in zero_cols:
                if matrix[row][col] == 0:
                    continue
                matrix[row][col] = 0


Solution = SolutionTwo


######################################################################################


inputs = []
inputs.append([[0]])
inputs.append([[1]])
inputs.append([[1, 0, 1], [1, 1, 1], [1, 1, 1]])
inputs.append([[0, 1, 2, 0], [3, 4, 5, 2], [1, 3, 1, 5]])


for idx, input in enumerate(inputs):

    matrix = deepcopy(input)

    print("####################\nExample %d:\n%s" % (idx, matrix_str(matrix)))
    Solution().setZeroes(matrix)

    print("Result:\n%s\n" % (matrix_str(matrix)))


print("\nALL DONE!\n")
