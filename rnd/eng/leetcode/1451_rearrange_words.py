"""

"""
from typing import List


class Solution:
    def arrangeWords(self, text: str) -> str:
        """
        Method: push the individual words into a dict, whose key is the length of the word. The value is a list
        Example: {2: ["on", "is"], 3: ["the"], 4: ["game"]}

        After that, one can sort the keys and spin out those words one by one

        """
        words = {}
        for m, w in enumerate(text.split(" ")):
            if m == 0:
                words[len(w)] = [w.lower()]
            else:
                words[len(w)] = words[len(w)] + [w] if len(w) in words else [w]

        idx = 0
        for k, v in sorted(words.items()):
            if idx == 0:
                v[0] = v[0].capitalize()
                new_text = " ".join(v)
            else:
                new_text = "%s %s" % (new_text, " ".join(v))
            idx += 1
        return new_text


######################################################################################


inputs = ["Leetcode is cool", "Keep calm and code on"]


for input in inputs:
    print("####################\nExample\n%s" % (str(input)))
    res = Solution().arrangeWords(input)

    print("Result = %s\n" % str(res))


print("ALL DONE!\n")
