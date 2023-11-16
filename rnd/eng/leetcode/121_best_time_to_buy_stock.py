"""
https://leetcode.com/problems/best-time-to-buy-and-sell-stock/

You are given an array prices where prices[i] is the price of a given stock on the ith day.

You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock.

Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return 0.

Constraints:

1 <= prices.length <= 10^5
0 <= prices[i] <= 10^4

# 
python eng/leetcode/121_best_time_to_buy_stock.py
"""
from typing import List


class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        max_profit = 0
        min_p = prices[0]
        for idx in range(1, len(prices)):
            p = prices[idx]
            if p - min_p > max_profit:
                max_profit = p - min_p
            if p < min_p:
                min_p = p

        max_profit = max(max_profit, 0)
        return max_profit


######################################################################################


inputs = {
    0: [7, 1, 5, 3, 6, 4],
    1: [7, 6, 4, 3, 1],
}


solution = Solution()
for idx, input in inputs.items():
    out = solution.maxProfit(input)
    print(f"idx: {idx}, out: {out}\n")
