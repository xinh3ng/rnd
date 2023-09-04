"""
https://leetcode.com/problems/letter-combinations-of-a-phone-number/description/

Constraints:

0 <= digits.length <= 4

digits[i] is a digit in the range ['2', '9'].

"""

from typing import List


class Solution:
    digits_to_letters = {
        "2": ["a", "b", "c"],
        "3": ["d", "e", "f"],
        "4": ["g", "h", "i"],
        "5": ["j", "k", "l"],
        "6": ["m", "n", "o"],
        "7": ["p", "q", "r", "s"],
        "8": ["t", "u", "v"],
        "9": ["w", "x", "y", "z"],
    }

    def letterCombinations(self, digits: str) -> List[str]:
        if digits is None or len(digits) == 0:
            return []

        def backtrack(combination, next_digits):
            if len(next_digits) >= 1:
                for letter in self.digits_to_letters[next_digits[0]]:
                    backtrack(combination + letter, next_digits[1:])
            else:
                output.append(combination)

        output = []
        backtrack("", digits)
        return output


######################################################################################


inputs = {
    "23": ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"],
    "": [],
}


solution = Solution()
for digits, out in inputs.items():
    assert "1" not in digits, f"digits: {digits} containt '1'"
    assert len(digits) <= 4

    out = solution.letterCombinations(digits)
    print(f"digits: {digits}, out: {out}\n")
