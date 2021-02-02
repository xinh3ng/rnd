"""
Merge k sorted linked lists and return it as one sorted list. Analyze and describe its complexity.

Example:

Input:
[
  1 -> 4 -> 5,
  1 -> 3 -> 4,
  2 -> 6
]
Output: 1 -> 1 -> 2 -> 3 -> 4 -> 4 -> 5 -> 6

"""
from typing import List


class ListNode:
    def __init__(self, val: int = 0, next: "ListNode" = None):
        self.val = val
        self.next = next

    def __str__(self):
        idx = 0
        while True:
            arr = str(self.val) if idx == 0 else arr + " -> " + str(self.val)
            self = self.next
            if self is None:
                break
            idx += 1

        return arr


def genListNode(arr):
    """Generate a ListNode instance from back to front"""
    idv_nodes = []
    for v in arr:
        idv_nodes.append(ListNode(v, None))  # There is no true "next" yet

    if len(idv_nodes) == 0:
        return

    # Walking from the last node to the first node
    for idx in range(len(idv_nodes) - 1, 0, -1):
        idv_nodes[idx - 1].next = idv_nodes[idx]  # Attach "next"

    return idv_nodes[0]


class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        if len(lists) == 0:
            return
        if all([v is None for v in lists]):
            return

        nodes = []
        while True:
            # Find the smallest number out of all smallest numbers
            values = [node.val for node in lists]
            min_idx, min_val = min(values)
            nodes.append(min_val)

            lists[min_idx] = lists[min_idx].next
            if lists[min_idx] is None:
                lists.pop(min_idx)

            if len(lists) == 0:
                break
        nodes = genListNode(nodes)
        return nodes


def min(values):
    min_val = values[0]
    min_idx = 0

    for idx, v in enumerate(values):
        if v < min_val:
            min_val = v
            min_idx = idx

    return min_idx, min_val


######################################################################################


inputs = {4: [[]], 3: [], 1: [[1]], 2: [[1], [1]], 0: [[1, 4, 5], [1, 3, 4], [2, 6]]}


for k, values in inputs.items():
    lists = [genListNode(v) for v in values]

    print("####################\nExample %d" % k)
    for listnode in lists:
        print(str(listnode))

    res = Solution().mergeKLists(lists)

    print("Result = %s\n" % res)


print("\nALL DONE!\n")
