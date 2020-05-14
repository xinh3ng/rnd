"""
"""


class Solution:
    next_steps = {
        0: (4, 6),
        1: (6, 8),
        2: (7, 9),
        3: (4, 8),
        4: (0, 3, 9),
        5: tuple(),
        6: (0, 1, 7),
        7: (2, 6),
        8: (1, 3),
        9: (2, 4),
    }

    # The key is "start_pos+num_hops"
    memorized_sequences = {}

    def knightDialer(self, N: int) -> int:
        """
        Args
            N: number of hops
        """
        count = 0
        for start_pos in range(10):
            res = self.gen_sequences(start_pos=start_pos, num_hops=N-1, sequence=[start_pos])
            self.memorized_sequences[f"{start_pos}+{N}"] = res

            res_list = list(res)
            print("%d:\n%s\n" %(start_pos, str(res_list)))
            count += len(res_list)

        print(f"count: {count}")
        return count

    def gen_sequences(self, start_pos: int, num_hops: int, sequence: list = None) -> int:
        """

        Args:
            sequence:
        """
        if f"{start_pos}+{num_hops}" in self.memorized_sequences:
            yield self.memorized_sequences.get(f"{start_pos}+{num_hops}")
            return

        sequence = [start_pos] if sequence is None else sequence
        if num_hops == 0:
            yield sequence
            return

        for s in self.next_steps[start_pos]:
            # s becomes the new starting position for the subsequent sequences
            yield from self.gen_sequences(s, num_hops - 1, sequence + [s])


######################################################################################


for n in [1, 2]:
    print("##########################")
    Solution().knightDialer(N=n)


print("\nALL DONE!\n")
