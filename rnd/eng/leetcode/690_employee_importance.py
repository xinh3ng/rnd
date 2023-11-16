"""Breadth first tree

You are given a data structure of employee information, which includes the employee's unique id, his importance value
and his direct subordinates' id.

For example, employee 1 is the leader of employee 2, and employee 2 is the leader of employee 3. They have importance
value 15, 10 and 5, respectively. Then employee 1 has a data structure like [1, 15, [2]], and employee 2 has
[2, 10, [3]], and employee 3 has [3, 5, []]. Note that although employee 3 is also a subordinate of employee 1,
the relationship is not direct.

Now given the employee information of a company, and an employee id, you need to return the total importance value of
this employee and all his subordinates.

Example 1:

Input: [[1, 5, [2, 3]], [2, 3, []], [3, 3, []]], 1
Output: 11
Explanation:

Employee 1 has importance value 5, and he has two direct subordinates: employee 2 and employee 3. They both have
importance value 3. So the total importance value of employee 1 is 5 + 3 + 3 = 11.

"""
from typing import List


class Employee:
    def __init__(self, id: int, importance: int, subordinates: List[int]):
        self.id = id
        self.importance = importance
        self.subordinates = subordinates


class Solution:
    def getImportance(self, employees: List["Employee"], id: int) -> int:
        emap = {x.id: x for x in employees}

        res = emap.get(id).importance  # The value of the current employee
        subordinates = emap.get(id).subordinates
        if len(subordinates) == 0:  # If he/she has no subord, return the importance value immediately
            return res
        return res + sum([self.getImportance(employees, sub) for sub in subordinates])


######################################################################################


inputs = {"0": ([[1, 5, [2, 3]], [2, 3, []], [3, 3, []]], 1), "1": ([[1, 5, [2, 3]], [2, 3, []], [3, 3, []]], 2)}


for idx, input in inputs.items():
    employees, id = input[0], input[1]

    print("####################\nExample %s:\n%s" % (idx, str(input)))
    employees = [Employee(x[0], x[1], x[2]) for x in employees]
    res = Solution().getImportance(employees, id=id)

    print(f"Result = {res}\n")


print("\nALL DONE!\n")
