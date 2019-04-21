from pdb import set_trace as debug
import sys

# sys.setrecursionlimit(10)


def fibo(n):
    assert n >= 1

    if n == 1:
        return 1
    if n == 2:
        return 1
    return fibo(n - 1) + fibo(n - 2)


def fibo_memo(n):
    """Fibonacci using memoization

    :param n:
    :return:
    """
    memory_map = {}
    assert n >= 1

    def helper(n):
        """helper function is the real fibonacci logic
        """
        if n in memory_map:
            return memory_map[n]

        if n == 1:
            memory_map[1] = 1
        elif n == 2:
            memory_map[2] = 1
        else:
            memory_map[n] = helper(n - 1) + helper(n - 2)

        return memory_map[n]

    return helper(n)


for n in [1, 2, 3, 5, 40]:
    print("fibo_memo(%3d) is: %d" % (n, fibo_memo(n)))

for n in [1, 2, 3, 5, 40]:
    print("fibo(%3d) is: %d" % (n, fibo(n)))

print("ALL DONE!")
