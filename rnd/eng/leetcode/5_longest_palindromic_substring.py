"""5. Longest Palindromic Substring
Medium

Given a string s, find the longest palindromic substring. You may assume that the maximum length of s is 1000.

Example 1:

Input: "babad"
Output: "bab"
Note: "aba" is also a valid answer.

Example 2:

Input: "cbbd"
Output: "bb"
"""
from typing import List


class SolutionOne:
    """"""

    def longestPalindrome(self, s: str) -> str:
        if not s:
            return ""

        N = len(s)

        # dp: 2D array. dp[l][r] records whether s[l] to s[r] is a palindrome
        dp = [[False] * N for _ in range(N)]
        for k in range(N):
            l, r = 0, k
            while r < N:
                if l == r:
                    dp[l][r] = True
                elif (s[l] == s[r]) and (r - l == 1):  # two adjacent letters are identical
                    dp[l][r] = True
                elif (s[l] == s[r]) and dp[l + 1][r - 1]:
                    dp[l][r] = True

                l, r = l + 1, r + 1

        # Get the longest palindrome
        res = ""
        for l in range(N):
            for r in range(l, N):
                if dp[l][r] and len(res) < (r - l + 1):  # Found a new and logner substring
                    res = s[l : (r + 1)]
        return res


class SolutionTwo:
    """
    https://www.youtube.com/watch?v=b4vgaENSRrY (good, I like his explanation)
    """

    def longestPalindrome(self, s: str) -> str:
        if not s:
            return ""

        # dp: 2D array. dp[l][r] records whether s[l] to s[r] is a palindrome
        N = len(s)
        dp = [[False] * N for _ in range(N)]

        # Case 1: start with one single element and expand both left and right
        for l in range(N):
            r = l
            while l >= 0 and r < N:
                if l == r:
                    dp[l][r] = True
                elif (s[l] == s[r]) and dp[l + 1][r - 1]:
                    dp[l][r] = True

                l, r = l - 1, r + 1

        # Case 2: start with two adjacent elements and expand both left and right
        for l in range(N - 1):
            r = l + 1
            if s[l] != s[r]:  # Adjacent elements not the same
                continue
            dp[l][r] = True
            while l >= 0 and r < N:
                if (s[l] == s[r]) and dp[l + 1][r - 1]:
                    dp[l][r] = True

                l, r = l - 1, r + 1

        # Finally, get the longest palindrome
        res = ""
        for l in range(N):
            for r in range(l, N):
                if dp[l][r] and len(res) < (r - l + 1):  # Found a new and logner substring
                    res = s[l : (r + 1)]
        return res


Solution = SolutionTwo


######################################################################################

inputs = {0: ("s", "s"), 1: ("babad", "bab"), 2: ("cbbd", "bb"), 3: ("baddad", "adda")}

for idx, input in inputs.items():
    string, answer = input[0], input[1]

    print("####################\nInput %d: '%s'" % (idx, string))
    res = Solution().longestPalindrome(s=string)
    print(f"Result: {res}, while expected: {answer}\n")

print("\nALL DONE!\n")
