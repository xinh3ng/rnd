"""
202. Happy Number

A happy number is a number defined by the following process: Starting with any positive integer, replace the number by the sum of the squares of its digits, and repeat the process until the number equals 1 (where it will stay), or it loops endlessly in a cycle which does not include 1. Those numbers for which this process ends in 1 are happy numbers.

Return True if n is a happy number, and False if not.
"""
from typing import List


def split_into_singles(n: int) -> List[int]:
    return [int(d) for d in str(n)]


class Solution:
    def isHappy(self, n: int) -> bool:
        while True:
            numbers = split_into_singles(n)
            # print(f"n = {n}, numbers = [{numbers}]")
            n = sum([x * x for x in numbers])
            if n == 1:
                return True
            if n == 4:
                return False


######################################################################################


inputs = [19, 2, 18]

for x in inputs:
    print(f"##########\nInput = {x}")
    res = Solution().isHappy(n=x)
    print(f"Input = {x}. Result = {res}\n")

print("\nALL DONE!\n")
