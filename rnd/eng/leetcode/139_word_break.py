"""
"""
from typing import List


class SolutionOne:
    """Use a resursive formula, keep skipping down the input word into shorter words

    It is too slow for the below case 3 (exponential growing new_s)
    """

    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        if s == "":
            return True

        new_s = self.get_partial_word(s, wordDict=wordDict)
        if len(new_s) == 0:
            return False

        return any([self.wordBreak(ns, wordDict) for ns in new_s])

    def get_partial_word(self, s: str, wordDict: List[str]) -> List[str]:
        new_s = []
        for w in wordDict:
            if w not in s:
                continue

            if w == s[0 : len(w)]:  # It is a match
                new_s.append(s[len(w) :])  # only keep the remainder part
        return new_s


class SolutionTwo:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        pass


Solution = SolutionOne


######################################################################################


inputs = {
    0: ["leetcode", ["leet", "code"]],
    1: ["applepenapple", ["apple", "pen"]],
    2: ["catsandog", ["cats", "dog", "sand", "and", "cat"]],
    3: [
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab",
        ["a", "aa", "aaa", "aaaa", "aaaaa", "aaaaaa", "aaaaaaa", "aaaaaaaa", "aaaaaaaaa", "aaaaaaaaaa"],
    ],
}


for idx, input in inputs.items():
    word, wordDict = input[0], input[1]

    print("####################\nExample %s:\n%s" % (idx, str(input)))
    res = Solution().wordBreak(s=word, wordDict=wordDict)
    print(f"Result = {res}\n")


print("\nALL DONE!\n")
