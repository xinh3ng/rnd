"""
Example 1:
Input: [2, 6, 4, 8, 10, 9, 15]
Output: 5
Explanation: You need to sort [6, 4, 8, 10, 9] in ascending order to make the whole array sorted in asc

"""
from typing import List


class Solution:
    def findUnsortedSubarray(self, nums: List[int]) -> int:
        length = len(nums)
        sorted_nums = sorted(nums)

        # Only recording the ones that have changed places
        diffs = [idx for idx in range(length) if nums[idx] != sorted_nums[idx]]
        if len(diffs) == 0:
            return 0
        return max(diffs) - min(diffs) + 1


######################################################################################


inputs = {"1": [2, 6, 4, 8, 10, 9, 15], "2": [3, 2, 1], "3": [1, 2, 3]}


for idx, nums in inputs.items():
    print("####################\nExample\n%s" % (str(nums)))
    res = Solution().findUnsortedSubarray(nums)

    print("Result = %s\n" % str(res))


print("ALL DONE!\n")
