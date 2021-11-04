"""

https://leetcode.com/problems/reverse-integer/

- It is considered medium on Leetcode

"""


class SolutionOne:
    def reverse(self, x: int) -> int:
        is_negative = x < 0
        x = x if x >= 0 else -x

        target = 0
        while x > 0:
            x, rem = divmod(x, 10)  # quotient and remainder
            target = target * 10 + rem
            if target > pow(2, 31) - 1 and (not is_negative):
                return 0
            if target > pow(2, 31) and is_negative:
                return 0

        target = -target if is_negative else target
        return target


Solution = SolutionOne


######################################################################################


examples = {1: (123, 321), 2: (-123, -321), 3: (120, 21), 4: (0, 0)}

for idx, example in examples.items():
    input, output = example[0], example[1]

    print("####################\nExample %s:\nInput  = %s" % (idx, str(input)))
    res = Solution().reverse(input)
    print(f"Result = {res}\n")
    assert res == output

print("\nALL DONE!\n")
