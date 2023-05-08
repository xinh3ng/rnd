"""Binary gap

Example 1:

Input: 22
Output: 2
Explanation:
22 in binary is 0b10110.
In the binary representation of 22, there are three ones, and two consecutive pairs of 1's.
The first consecutive pair of 1's have distance 2.
The second consecutive pair of 1's have distance 1.
The answer is the largest of these two distances, which is 2.
Example 2:

Input: 5
Output: 2
Explanation:
5 in binary is 0b101.
Example 3:

Input: 6
Output: 1
Explanation:
6 in binary is 0b110.
Example 4:

Input: 8
Output: 0
Explanation:
8 in binary is 0b1000.
There aren't any consecutive pairs of 1's in the binary representation of 8, so we return 0.

"""


class Solution:
    def binaryGap(self, N: int) -> int:
        # Converting to a binary number to go though 1 by 1
        gaps = []
        start, end = None, None
        idx = -1
        for x in str(bin(N)):
            idx += 1
            if x != "1":
                continue

            if start is None:
                start = idx

            else:  # Calculate the distance. And the end becomes the new start
                gaps.append(idx - start)
                start = idx

        # print(str(gaps))
        result = max(gaps) if len(gaps) >= 1 else 0
        return result


######################################################################################


inputs = [22, 8, 5, 6]


for idx, N in enumerate(inputs):
    print("####################\nExample %s: %s" % (idx, str(N)))
    res = Solution().binaryGap(N)
    print(f"Result = {res}\n")


print("\nALL DONE!\n")
