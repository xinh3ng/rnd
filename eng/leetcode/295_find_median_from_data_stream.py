"""
Median is the middle value in an ordered integer list. If the size of the list is even, there is no middle value. So the median is the mean of the two middle value.
For example,

[2,3,4], the median is 3

[2,3], the median is (2 + 3) / 2 = 2.5

Design a data structure that supports the following two operations:

    void addNum(int num) - Add a integer number from the data stream to the data structure.
    double findMedian() - Return the median of all elements so far.

Example:

addNum(1)
addNum(2)
findMedian() -> 1.5
addNum(3) 
findMedian() -> 2
"""

from pdb import set_trace as debug
import bisect


class MedianFinder(object):
    def __init__(self):
        """
        initialize your data structure here.
        """
        self.values = []

    def addNum(self, num):
        # The bisect module implements an algorithm for inserting elements into a list while maintaining the list in sorted order.
        bisect.insort(self.values, num)

    def findMedian(self):
        values_length = len(self.values)
        index = values_length // 2

        if values_length % 2 == 0:
            return (self.values[index - 1] + self.values[index]) / 2
        else:
            return float(self.values[index])


# Your MedianFinder object will be instantiated and called as such:
# ds: Data Stream
ds = MedianFinder()
ds.addNum(1)
print(ds.findMedian())

ds.addNum(2)
print(ds.findMedian())

ds.addNum(3)
print(ds.findMedian())

# Another example
print("")
ds = MedianFinder()
for x in [-1, -2, -3, -4, -5]:
    ds.addNum(x)
    print(ds.findMedian())
