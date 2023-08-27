"""
"""
from typing import List


class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        return


######################################################################################


inputs = {
    "23": ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"],
    "": [],
}


solution = Solution()
for digits, out in inputs.items():
    print("####################")

    out = solution.letterCombinations(digits)
    print(f"digits: {digits}, out: {out}")
