"""
https://leetcode.com/problems/product-of-array-except-self/

Given an integer array nums, return an array answer such that answer[i] is equal to the product of all the elements of nums except nums[i].

The product of any prefix or suffix of nums is guaranteed to fit in a 32-bit integer.

You must write an algorithm that runs in O(n) time and without using the division operation.

Example 1:

Input: nums = [1,2,3,4]
Output: [24,12,8,6]
Example 2:

Input: nums = [-1,1,0,-3,3]
Output: [0,0,9,0,0]

# 
python eng/leetcode/121_best_time_to_buy_stock.py
"""
from typing import List


class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:

        output = [1 for _ in nums]

        for idx, n in enumerate(nums):
            for x in range(len(output)):
                if x == idx:
                    continue
                output[x] *= n

        return output


######################################################################################


data = [([1, 2, 3, 4], [24, 12, 8, 6]), ([-1, 1, 0, -3, 3], [0, 0, 9, 0, 0])]

solution = Solution()
for idx, (input, output) in enumerate(data):
    out = solution.productExceptSelf(input)
    print(f"idx: {idx}, out: {out}\n")
