"""

https://leetcode.com/problems/search-in-rotated-sorted-array/

- It is considered medium on Leetcode

- Find the pivot and then it is easy to decide if I should focus on the left or right half

"""
from typing import List


class SolutionOne:
    def search(self, nums: List[int], target: int) -> int:
        n = len(nums)
        piv = self.search_pivot(nums)

        if target == nums[0]:
            return 0
        if target == nums[-1]:
            return n - 1
        if target == nums[piv]:
            return piv
        if target == nums[piv - 1]:
            return piv - 1

        if (target > nums[piv]) and (target < nums[-1]):  # If in the 2nd half
            res = self.binary_search(nums[piv:], target=target)
            if res == -1:
                return -1
            else:
                return piv + res
        else:  # In the 2nd half
            return self.binary_search(nums[:piv], target=target)

        return -1

    def search_pivot(self, nums: List[int]):
        """Find the location (index) of the pivot"""

        pivot = 0
        for idx, val in enumerate(nums):
            if idx == 0:
                prev_val = val
                continue
            if val < prev_val:
                pivot = idx
                break
        return pivot

    def binary_search(self, nums: List[int], target: int) -> int:
        """Binary search of a sorted integer array"""
        low = 0
        high = len(nums) - 1
        mid = (high + low) // 2
        while low <= high:
            if nums[mid] < target:  # If x is greater, ignore left half
                low = mid + 1
            elif nums[mid] > target:  # If x is smaller, ignore right half
                high = mid - 1
            else:  # x is present at mid
                return mid
            mid = (high + low) // 2
        # If we reach here, then the element was not present
        return -1


Solution = SolutionOne


######################################################################################


examples = {
    6: {"nums": [1, 2, 3, 4, 5, 6], "target": 4, "output": 3},
    4: {
        "nums": [4, 5, 6, 7, 0, 1, 2],
        "target": 3,
        "output": -1,
    },
    1: {
        "nums": [4, 5, 6, 7, 0, 1, 2, 3],
        "target": 0,
        "output": 4,
    },
    2: {
        "nums": [4, 5, 6, 7, 0, 1, 2, 3],
        "target": 2,
        "output": 6,
    },
    3: {
        "nums": [4, 5, 6, 7, 0, 1, 2, 3],
        "target": 6,
        "output": 2,
    },
    5: {
        "nums": [1],
        "target": 0,
        "output": -1,
    },
}

for idx, example in examples.items():
    nums, target, output = example["nums"], example["target"], example["output"]

    print("\n####################\nExample %s:\nnums: %s, target: %d, solution: %d" % (idx, str(nums), target, output))
    res = Solution().search(nums=nums, target=target)
    print(f"-> result = {res}\n")
    assert res == output

print("\nALL DONE!\n")
