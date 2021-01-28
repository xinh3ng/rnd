class SolutionOne:
    """"""

    def myAtoi(self, s: str) -> bool:

        leading_spaces_processed = False
        is_negative = False
        result = 0
        for char in s:
            if not leading_spaces_processed:  # If there may be still leading spaces
                if char == " ":
                    continue
                elif char == "-":  # This indicates the leading sign is a negative sign
                    is_negative = True
                    leading_spaces_processed = True
                    continue
                elif char == "+":  # This indicates the leading sign is a positive sign
                    leading_spaces_processed = True
                    continue
                else:
                    leading_spaces_processed = True

            try:
                result = result * 10 + int(char)
            except Exception as e:
                # If can't process this char, the process should end
                break

        result = -1 * result if is_negative else result
        return result


Solution = SolutionOne


######################################################################################


examples = {
    0: ("42", 42),
    1: ("   -42", -42),
    2: ("4193 with words", 4193),
    3: ("words and 987", 0),
    4: (" +42", 42),
    5: (" 42+", 42),
    6: (" 42-", 42),
    7: ("-91283472332", -2147483648),
    8: ("  -91283472332+", -2147483648),
}

for idx, example in examples.items():
    input, output = example[0], example[1]

    print("####################\nExample %s:\nInput  = %s" % (idx, str(input)))
    res = Solution().myAtoi(input)
    print(f"Result = {res}\n")
    assert res == output

print("\nALL DONE!\n")
