"""
https://leetcode.com/problems/median-of-two-sorted-arrays/

- It is considered HARD on Leetcode

"""
from typing import List


class SolutionOne:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        return


Solution = SolutionOne


######################################################################################


examples = {
    1: {
        "nums1": [1, 3],
        "nums2": [2],
        "output": 2.00,
    },
    2: {
        "nums1": [1, 2],
        "nums2": [3, 4],
        "output": 2.50,
    },
    3: {
        "nums1": [0, 0],
        "nums2": [0, 0],
        "output": 0.00,
    },
    4: {
        "nums1": [],
        "nums2": [1],
        "output": 1.00,
    },
    5: {
        "nums1": [2],
        "nums2": [],
        "output": 2.00,
    },
}


for idx, example in examples.items():
    nums, target, output = example["nums"], example["target"], example["output"]

    print("\n####################\nExample %s:\nnums: %s, target: %d, solution: %d" % (idx, str(nums), target, output))
    res = Solution().search(nums=nums, target=target)
    print(f"-> result = {res}\n")
    assert res == output

print("\nALL DONE!\n")
