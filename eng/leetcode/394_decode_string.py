"""
Given an encoded string, return it's decoded string.

The encoding rule is: k[encoded_string], where the encoded_string inside the square brackets is being repeated exactly k times. Note that k is guaranteed to be a positive integer.

You may assume that the input string is always valid; No extra white spaces, square brackets are well-formed, etc.

Furthermore, you may assume that the original data does not contain any digits and that digits are only for those repeat numbers, k. For example, there won't be input like 3a or 2[4].

Examples:

s = "3[a]2[bc]", return "aaabcbc".
s = "3[a2[c]]", return "accaccacc".
s = "2[abc]3[cd]ef", return "abcabccdcdcdef".
"""

from pdb import set_trace as debug


class SolutionOne(object):
    """
    https://leetcode.com/problems/decode-string/discuss/87563/Share-my-Python-Stack-Simple-Solution-(Easy-to-understand)
    """

    def decodeString(self, s):
        stack = []
        stack.append(["", 1])
        num = ""
        for ch in s:
            if ch.isdigit():
              num += ch
            elif ch == '[':
                stack.append(["", int(num)])
                num = ""
            elif ch == ']':
                st, k = stack.pop()
                stack[-1][0] += st*k
            else:
                stack[-1][0] += ch
        return stack[0][0]


class Solution(object):
    """
    Keep translating the left most [ and its enclosure. During the process, I need to 
    - identify location of the pairing ]
    - identify the proceeeding digits
    """
    def decodeString(self, s):
        text = s
        print("\n" + text)
        while "[" in text:
            num = ""
            pos_num = None  # Starting position of a numerical
            for pos, ch in enumerate(text):
                if ch in "0123456789":
                    num = num + ch
                    if pos_num is None:  # Record the starting position of a numerical
                        pos_num = pos
                elif ch == "[":
                    pos_right_bracket = self.locate_right_bracket(text, pos+1)
                    replacement = text[(pos+1):pos_right_bracket] * int(num)
                    text = text[0:pos_num] + replacement + text[(pos_right_bracket+1):] 
                    print(text)
                    break  # Exit the for loop, i.e. return to the while loop
        return text


    def locate_right_bracket(self, text, start_pos=0):
        """
        """
        cnt_left_bracket = 0
        for pos, ch in enumerate(text):
            if pos < start_pos:  # Skip if pos is less than the starting position
                continue
            if ch == "[":
                cnt_left_bracket += 1

            if ch == "]": 
                if cnt_left_bracket > 0:
                    cnt_left_bracket -= 1
                else:  # The return condition: ch == "]" and count of left bracket is 0
                    return pos
        raise AssertionError("Should not reach this place")

########################################

for s, answer in [
          ("10[a]", "aaaaaaaaaa"),
          ("3[a]2[bc]", "aaabcbc"),
	      ("3[a2[c]]",  "accaccacc"),
	      ("2[abc]3[cd]ef",  "abcabccdcdcdef")
	      ]:
	result = Solution().decodeString(s)
	
	print("%-20s: %s" % (s, result))
	assert result == answer

print("ALL DONE!\n")


