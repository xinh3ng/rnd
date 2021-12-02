"""
https://leetcode.com/problems/design-add-and-search-words-data-structure/

- It is considered medium on Leetcode

# Your WordDictionary object will be instantiated and called as such:
# obj = WordDictionary()
# obj.addWord(word)
# param_2 = obj.search(word)
"""
from typing import List


class WordDictionary:
    """
    After:
    d = WordDictionary()
    d.addWord("bad")
    d.addWord("dad")
    d.addWord("mad")
    d.addWord("a")
    d.addWord("ab")

    It became:

    Print self.map:
    {'0-b': {'bad'}, '1-a': {'bad'}, '2-d': {'bad'}}

    Print self.map:
    {'0-b': {'bad'}, '1-a': {'dad', 'bad'}, '2-d': {'dad', 'bad'}, '0-d': {'dad'}}

    Print self.map:
    {'0-b': {'bad'}, '1-a': {'dad', 'mad', 'bad'}, '2-d': {'dad', 'mad', 'bad'}, '0-d': {'dad'}, '0-m': {'mad'}}

    Print self.map:
    {'0-b': {'bad'}, '1-a': {'dad', 'mad', 'bad'}, '2-d': {'dad', 'mad', 'bad'}, '0-d': {'dad'}, '0-m': {'mad'}, '0-a': {'a'}}

    Print self.map:
    {'0-b': {'bad'}, '1-a': {'dad', 'mad', 'bad'}, '2-d': {'dad', 'mad', 'bad'}, '0-d': {'dad'}, '0-m': {'mad'}, '0-a': {'ab', 'a'}, '1-b': {'ab'}}

    """

    def __init__(self):
        self.map = {}

    def addWord(self, word: str) -> None:
        for i, x in enumerate(word):
            key = str(i) + "-" + x
            if key not in self.map:
                self.map[key] = set([word])
            else:
                self.map[key].add(word)
        print("Print self.map:")
        print(self.map)

    def search(self, word: str) -> bool:

        candidates = None
        for i, x in enumerate(word):
            if x == ".":
                continue
            key = str(i) + "-" + x
            if key not in self.map:
                return False
            else:  # found the letter at the right location
                if candidates is None:
                    candidates = self.map[key]
                else:
                    candidates = candidates.intersection(self.map[key])

            if len(candidates) == 0:
                return False

        if candidates is None:
            return True

        for d in candidates:
            if len(word) == len(d):
                return True
        return False


######################################################################################

d = WordDictionary()
d.addWord("bad")
d.addWord("dad")
d.addWord("mad")
d.addWord("a")
d.addWord("ab")

res = d.search("a.")
assert res

res = d.search(".")
assert res

res = d.search("b...")
assert not res

res = d.search("pad")
assert not res

res = d.search("bad")
assert res

res = d.search(".ad")
assert res

res = d.search("b..")
assert res

print("\nALL DONE!\n")
