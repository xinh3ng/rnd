"""
"""
from typing import List


class Solution:
    """
    | arr1[i] - arr1[j] | + | arr2[i] - arr2[j] | + | i - j |
    """

    def maxAbsValExpr(self, arr1: List[int], arr2: List[int]) -> int:
        return self.maxAbsValExprTwo(arr1, arr2)

    def maxAbsValExprOne(self, arr1: List[int], arr2: List[int]) -> int:
        """
        Go through 2 "for" loops, O(n^2). May not be accepted by leetcode
        """
        length = len(arr1)
        max_val = 0
        for x in range(length):
            for y in range(length):
                if y <= x:
                    continue

                val = abs(arr1[y] - arr1[x]) + abs(arr2[y] - arr2[x]) + abs(y - x)
                max_val = max(val, max_val)
        return max_val

    def maxAbsValExprTwo(self, arr1: List[int], arr2: List[int]) -> int:
        """

        One "for" loop, the observation is there are max four situations
        """
        x0, x1, x2, x3 = [], [], [], []
        for m in range(len(arr1)):
            x0.append(arr1[m] + arr2[m] + m)
            x1.append(arr1[m] - arr2[m] + m)
            x2.append(-arr1[m] + arr2[m] + m)
            x3.append(-arr1[m] - arr2[m] + m)

        max_val = max(max(x0) - min(x0), max(x1) - min(x1), max(x2) - min(x2), max(x3) - min(x3))

        return max_val


######################################################################################


inputs = {"0": ([1, 2, 3, 4], [-1, 4, 5, 6]), "1": ([1, -2, -5, 0, 10], [0, -2, -1, -7, -4])}


for idx, input in inputs.items():
    arr1, arr2 = input[0], input[1]

    print("####################\nExample %s:\n%s" % (idx, str(input)))
    res = Solution().maxAbsValExpr(arr1, arr2)

    print("Result = %d\n" % res)


print("\nALL DONE!\n")
