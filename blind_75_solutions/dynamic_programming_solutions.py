"""
Module containing dynamic programming-related solutions for Blind 75 LeetCode questions.
"""

from typing import List, Optional
from blind_75_solutions import Solution


class DynamicProgrammingSolutions:
    """Class containing dynamic programming-related solutions for Blind 75 LeetCode questions."""

    @staticmethod
    def climbing_stairs() -> Solution:
        """
        Generate a Solution object for the Climbing Stairs problem.

        Returns:
            Solution: A Solution object containing details of the Climbing Stairs problem.
        """

        def problem_statement() -> str:
            return """You are climbing a staircase. It takes n steps to reach the top.

Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?

 

Example 1:

Input: n = 2
Output: 2
Explanation: There are two ways to climb to the top.
1. 1 step + 1 step
2. 2 steps

Example 2:

Input: n = 3
Output: 3
Explanation: There are three ways to climb to the top.
1. 1 step + 1 step + 1 step
2. 1 step + 2 steps
3. 2 steps + 1 step

 

Constraints:

    1 <= n <= 45

https://leetcode.com/problems/climbing-stairs/description/
"""

        def easy_solution(n: int) -> int:
            # Dynamic programming solution with O(n) space complexity
            if n <= 2:
                return n
            dp = [0] * (n + 1)
            dp[1] = 1
            dp[2] = 2
            for i in range(3, n + 1):
                dp[i] = dp[i - 1] + dp[i - 2]
            return dp[n]

        def optimized_solution(n: int) -> int:
            # Optimized solution with O(1) space complexity
            if n <= 2:
                return n
            a, b = 1, 2
            for _ in range(3, n + 1):
                a, b = b, a + b
            return b

        return Solution(
            question="Climbing Stairs",
            easy_solution=easy_solution,
            optimized_solution=optimized_solution,
            time_complexity="O(n)",
            space_complexity="O(1)",
            similar_questions=[
                "Min Cost Climbing Stairs",
                "Fibonacci Number",
                "N-th Tribonacci Number",
            ],
            problem_statement=problem_statement,
        )

    @staticmethod
    def coin_change() -> Solution:
        """
        Generate a Solution object for the Coin Change problem.

        Returns:
            Solution: A Solution object containing details of the Coin Change problem.
        """

        def problem_statement() -> str:
            return """You are given an integer array coins representing coins of different denominations and an integer amount representing a total amount of money.

Return the fewest number of coins that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return -1.

You may assume that you have an infinite number of each kind of coin.

 

Example 1:

Input: coins = [1,2,5], amount = 11
Output: 3
Explanation: 11 = 5 + 5 + 1

Example 2:

Input: coins = [2], amount = 3
Output: -1

Example 3:

Input: coins = [1], amount = 0
Output: 0

 

Constraints:

    1 <= coins.length <= 12
    1 <= coins[i] <= 231 - 1
    0 <= amount <= 104

https://leetcode.com/problems/coin-change/description/
"""

        def easy_solution(coins: List[int], amount: int) -> int:
            # Dynamic programming solution with O(amount * len(coins)) time complexity
            dp = [float("inf")] * (amount + 1)
            dp[0] = 0
            for coin in coins:
                for x in range(coin, amount + 1):
                    dp[x] = min(dp[x], dp[x - coin] + 1)
            return dp[amount] if dp[amount] != float("inf") else -1

        def optimized_solution(coins: List[int], amount: int) -> int:
            # Optimized dynamic programming solution
            dp = [float("inf")] * (amount + 1)
            dp[0] = 0
            for i in range(1, amount + 1):
                dp[i] = (
                    min(dp[i - c] if i - c >= 0 else float("inf") for c in coins) + 1
                )
            return dp[amount] if dp[amount] != float("inf") else -1

        return Solution(
            question="Coin Change",
            easy_solution=easy_solution,
            optimized_solution=optimized_solution,
            time_complexity="O(amount * len(coins))",
            space_complexity="O(amount)",
            similar_questions=["Minimum Cost For Tickets"],
            problem_statement=problem_statement,
        )

    @staticmethod
    def longest_increasing_subsequence() -> Solution:
        """
        Generate a Solution object for the Longest Increasing Subsequence problem.

        Returns:
            Solution: A Solution object containing details of the Longest Increasing Subsequence problem.
        """

        def problem_statement() -> str:
            return """Given an integer array nums, return the length of the longest strictly increasing subsequence.

 

Example 1:

Input: nums = [10,9,2,5,3,7,101,18]
Output: 4
Explanation: The longest increasing subsequence is [2,3,7,101], therefore the length is 4.

Example 2:

Input: nums = [0,1,0,3,2,3]
Output: 4

Example 3:

Input: nums = [7,7,7,7,7,7,7]
Output: 1

 

Constraints:

    1 <= nums.length <= 2500
    -104 <= nums[i] <= 104

 

Follow up: Can you come up with an algorithm that runs in O(n log(n)) time complexity?

https://leetcode.com/problems/longest-increasing-subsequence/description/
"""

        def easy_solution(nums: List[int]) -> int:
            # Dynamic programming solution with O(n^2) time complexity
            if not nums:
                return 0
            dp = [1] * len(nums)
            for i in range(1, len(nums)):
                for j in range(i):
                    if nums[i] > nums[j]:
                        dp[i] = max(dp[i], dp[j] + 1)
            return max(dp)

        def optimized_solution(nums: List[int]) -> int:
            # Optimized solution using binary search with O(n log n) time complexity
            tails = [0] * len(nums)
            size = 0
            for num in nums:
                i, j = 0, size
                while i != j:
                    m = (i + j) // 2
                    if tails[m] < num:
                        i = m + 1
                    else:
                        j = m
                tails[i] = num
                size = max(i + 1, size)
            return size

        return Solution(
            question="Longest Increasing Subsequence",
            easy_solution=easy_solution,
            optimized_solution=optimized_solution,
            time_complexity="O(n log n)",
            space_complexity="O(n)",
            similar_questions=[
                "Increasing Triplet Subsequence",
                "Russian Doll Envelopes",
                "Maximum Length of Pair Chain",
            ],
            problem_statement=problem_statement,
        )

    @staticmethod
    def word_break() -> Solution:
        """
        Generate a Solution object for the Word Break problem.

        Returns:
            Solution: A Solution object containing details of the Word Break problem.
        """

        def problem_statement() -> str:
            return """Given a string s and a dictionary of strings wordDict, return true if s can be segmented into a space-separated sequence of one or more dictionary words.

Note that the same word in the dictionary may be reused multiple times in the segmentation.

 

Example 1:

Input: s = "leetcode", wordDict = ["leet","code"]
Output: true
Explanation: Return true because "leetcode" can be segmented as "leet code".

Example 2:

Input: s = "applepenapple", wordDict = ["apple","pen"]
Output: true
Explanation: Return true because "applepenapple" can be segmented as "apple pen apple".
Note that you are allowed to reuse a dictionary word.

Example 3:

Input: s = "catsandog", wordDict = ["cats","dog","sand","and","cat"]
Output: false

 

Constraints:

    1 <= s.length <= 300
    1 <= wordDict.length <= 1000
    1 <= wordDict[i].length <= 20
    s and wordDict[i] consist of only lowercase English letters.
    All the strings of wordDict are unique.

https://leetcode.com/problems/word-break/description/
"""

        def easy_solution(s: str, wordDict: List[str]) -> bool:
            # Dynamic programming solution with O(n^2) time complexity
            dp = [False] * (len(s) + 1)
            dp[0] = True
            for i in range(1, len(s) + 1):
                for j in range(i):
                    if dp[j] and s[j:i] in wordDict:
                        dp[i] = True
                        break
            return dp[len(s)]

        def optimized_solution(s: str, wordDict: List[str]) -> bool:
            # Optimized dynamic programming solution using a set for wordDict
            word_set = set(wordDict)
            dp = [False] * (len(s) + 1)
            dp[0] = True
            for i in range(1, len(s) + 1):
                for j in range(i):
                    if dp[j] and s[j:i] in word_set:
                        dp[i] = True
                        break
            return dp[len(s)]

        return Solution(
            question="Word Break",
            easy_solution=easy_solution,
            optimized_solution=optimized_solution,
            time_complexity="O(n^2)",
            space_complexity="O(n)",
            similar_questions=["Word Break II"],
            problem_statement=problem_statement,
        )

    @staticmethod
    def combination_sum() -> Solution:
        """
        Generate a Solution object for the Combination Sum problem.

        Returns:
            Solution: A Solution object containing details of the Combination Sum problem.
        """

        def problem_statement() -> str:
            return """Given an array of distinct integers candidates and a target integer target, return a list of all unique combinations of candidates where the chosen numbers sum to target. You may return the combinations in any order.

The same number may be chosen from candidates an unlimited number of times. Two combinations are unique if the frequency of at least one of the chosen numbers is different.

The test cases are generated such that the number of unique combinations that sum up to target is less than 150 combinations for the given input.

 

Example 1:

Input: candidates = [2,3,6,7], target = 7
Output: [[2,2,3],[7]]
Explanation:
2 and 3 are the only numbers that can be combined to get 7.
2 can be used twice, and the combination [2,2,3] is valid.
7 can be used once, and the combination [7] is valid.

Example 2:

Input: candidates = [2,3,5], target = 8
Output: [[2,2,2,2],[2,3,3],[3,5]]

Example 3:

Input: candidates = [2], target = 1
Output: []

 

Constraints:

    1 <= candidates.length <= 30
    2 <= candidates[i] <= 40
    All elements of candidates are distinct.
    1 <= target <= 40

https://leetcode.com/problems/combination-sum/description/
"""

        def easy_solution(candidates: List[int], target: int) -> List[List[int]]:
            # Backtracking solution
            def backtrack(start: int, target: int, path: List[int]) -> None:
                if target == 0:
                    result.append(path[:])
                    return
                for i in range(start, len(candidates)):
                    if candidates[i] > target:
                        break
                    path.append(candidates[i])
                    backtrack(i, target - candidates[i], path)
                    path.pop()

            result: List[List[int]] = []
            candidates.sort()
            backtrack(0, target, [])
            return result

        def optimized_solution(candidates: List[int], target: int) -> List[List[int]]:
            # Dynamic programming solution
            dp: List[List[List[int]]] = [[] for _ in range(target + 1)]
            dp[0] = [[]]

            for c in candidates:
                for i in range(c, target + 1):
                    dp[i].extend([comb + [c] for comb in dp[i - c]])

            return dp[target]

        return Solution(
            question="Combination Sum",
            easy_solution=easy_solution,
            optimized_solution=optimized_solution,
            time_complexity="O(n * target)",
            space_complexity="O(target)",
            similar_questions=[
                "Letter Combinations of a Phone Number",
                "Combination Sum II",
                "Combinations",
                "Combination Sum III",
                "Factor Combinations",
                "Combination Sum IV",
            ],
            problem_statement=problem_statement,
        )

    @staticmethod
    def house_robber() -> Solution:
        """
        Generate a Solution object for the House Robber problem.

        Returns:
            Solution: A Solution object containing details of the House Robber problem.
        """

        def problem_statement() -> str:
            return """You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed, the only constraint stopping you from robbing each of them is that adjacent houses have security systems connected and it will automatically contact the police if two adjacent houses were broken into on the same night.

Given an integer array nums representing the amount of money of each house, return the maximum amount of money you can rob tonight without alerting the police.

 

Example 1:

Input: nums = [1,2,3,1]
Output: 4
Explanation: Rob house 1 (money = 1) and then rob house 3 (money = 3).
Total amount you can rob = 1 + 3 = 4.

Example 2:

Input: nums = [2,7,9,3,1]
Output: 12
Explanation: Rob house 1 (money = 2), rob house 3 (money = 9), and rob house 5 (money = 1).
Total amount you can rob = 2 + 9 + 1 = 12.

 

Constraints:

    1 <= nums.length <= 100
    0 <= nums[i] <= 400

https://leetcode.com/problems/house-robber/description/
"""

        def easy_solution(nums: List[int]) -> int:
            # Dynamic programming solution with O(n) space complexity
            if not nums:
                return 0
            if len(nums) <= 2:
                return max(nums)
            dp = [0] * len(nums)
            dp[0] = nums[0]
            dp[1] = max(nums[0], nums[1])
            for i in range(2, len(nums)):
                dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])
            return dp[-1]

        def optimized_solution(nums: List[int]) -> int:
            # Optimized solution with O(1) space complexity
            if not nums:
                return 0
            prev, curr = 0, 0
            for num in nums:
                prev, curr = curr, max(curr, prev + num)
            return curr

        return Solution(
            question="House Robber",
            easy_solution=easy_solution,
            optimized_solution=optimized_solution,
            time_complexity="O(n)",
            space_complexity="O(1)",
            similar_questions=[
                "Maximum Product Subarray",
                "House Robber II",
                "Paint House",
                "Paint Fence",
                "House Robber III",
            ],
            problem_statement=problem_statement,
        )

    @staticmethod
    def house_robber_ii() -> Solution:
        """
        Generate a Solution object for the House Robber II problem.

        Returns:
            Solution: A Solution object containing details of the House Robber II problem.
        """

        def problem_statement() -> str:
            return """You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed. All houses at this place are arranged in a circle. That means the first house is the neighbor of the last one. Meanwhile, adjacent houses have a security system connected, and it will automatically contact the police if two adjacent houses were broken into on the same night.

Given an integer array nums representing the amount of money of each house, return the maximum amount of money you can rob tonight without alerting the police.

 

Example 1:

Input: nums = [2,3,2]
Output: 3
Explanation: You cannot rob house 1 (money = 2) and then rob house 3 (money = 2), because they are adjacent houses.

Example 2:

Input: nums = [1,2,3,1]
Output: 4
Explanation: Rob house 1 (money = 1) and then rob house 3 (money = 3).
Total amount you can rob = 1 + 3 = 4.

Example 3:

Input: nums = [1,2,3]
Output: 3

 

Constraints:

    1 <= nums.length <= 100
    0 <= nums[i] <= 1000

https://leetcode.com/problems/house-robber-ii/description/
"""

        def rob(nums: List[int]) -> int:
            def simple_rob(nums: List[int]) -> int:
                prev, curr = 0, 0
                for num in nums:
                    prev, curr = curr, max(curr, prev + num)
                return curr

            if len(nums) == 1:
                return nums[0]
            return max(simple_rob(nums[:-1]), simple_rob(nums[1:]))

        def easy_solution(nums: List[int]) -> int:
            return rob(nums)

        def optimized_solution(nums: List[int]) -> int:
            return rob(nums)

        return Solution(
            question="House Robber II",
            easy_solution=easy_solution,
            optimized_solution=optimized_solution,
            time_complexity="O(n)",
            space_complexity="O(1)",
            similar_questions=[
                "House Robber",
                "Paint House",
                "Paint Fence",
                "House Robber III",
                "Non-negative Integers without Consecutive Ones",
            ],
            problem_statement=problem_statement,
        )

    @staticmethod
    def decode_ways() -> Solution:
        """
        Generate a Solution object for the Decode Ways problem.

        Returns:
            Solution: A Solution object containing details of the Decode Ways problem.
        """

        def problem_statement() -> str:
            return """A message containing letters from A-Z can be encoded into numbers using the following mapping:

'A' -> "1"
'B' -> "2"
...
'Z' -> "26"

To decode an encoded message, all the digits must be grouped then mapped back into letters using the reverse of the mapping above (there may be multiple ways). 

For example, "11106" can be mapped into:

    "AAJF" with the grouping (1 1 10 6)
    "KJF" with the grouping (11 10 6)

Note that the grouping (1 11 06) is invalid because "06" cannot be mapped into 'F' since "6" is different from "06".

Given a string s containing only digits, return the number of ways to decode it.

The test cases are generated so that the answer fits in a 32-bit integer.

 

Example 1:

Input: s = "12"
Output: 2
Explanation: "12" could be decoded as "AB" (1 2) or "L" (12).

Example 2:

Input: s = "226"
Output: 3
Explanation: "226" could be decoded as "BZ" (2 26), "VF" (22 6), or "BBF" (2 2 6).

Example 3:

Input: s = "06"
Output: 0
Explanation: "06" cannot be mapped to "F" because of the leading zero ("6" is different from "06").

 

Constraints:

    1 <= s.length <= 100
    s contains only digits and may contain leading zero(s).

https://leetcode.com/problems/decode-ways/description/
"""

        def easy_solution(s: str) -> int:
            # Dynamic programming solution with O(n) space complexity
            if not s or s[0] == "0":
                return 0
            dp = [0] * (len(s) + 1)
            dp[0] = 1
            dp[1] = 1
            for i in range(2, len(s) + 1):
                if s[i - 1] != "0":
                    dp[i] += dp[i - 1]
                if s[i - 2] == "1" or (s[i - 2] == "2" and s[i - 1] <= "6"):
                    dp[i] += dp[i - 2]
            return dp[-1]

        def optimized_solution(s: str) -> int:
            # Optimized solution with O(1) space complexity
            if not s or s[0] == "0":
                return 0
            prev, curr = 1, 1
            for i in range(1, len(s)):
                temp = 0
                if s[i] != "0":
                    temp = curr
                if s[i - 1] == "1" or (s[i - 1] == "2" and s[i] <= "6"):
                    temp += prev
                prev, curr = curr, temp
            return curr

        return Solution(
            question="Decode Ways",
            easy_solution=easy_solution,
            optimized_solution=optimized_solution,
            time_complexity="O(n)",
            space_complexity="O(1)",
            similar_questions=["Decode Ways II"],
            problem_statement=problem_statement,
        )

    @staticmethod
    def unique_paths() -> Solution:
        """
        Generate a Solution object for the Unique Paths problem.

        Returns:
            Solution: A Solution object containing details of the Unique Paths problem.
        """

        def problem_statement() -> str:
            return """There is a robot on an m x n grid. The robot is initially located at the top-left corner (i.e., grid[0][0]). The robot tries to move to the bottom-right corner (i.e., grid[m-1][n-1]). The robot can only move either down or right at any point in time.

Given the two integers m and n, return the number of possible unique paths that the robot can take to reach the bottom-right corner.

The test cases are generated so that the answer will be less than or equal to 2 * 109.

 

Example 1:

Input: m = 3, n = 7
Output: 28

Example 2:

Input: m = 3, n = 2
Output: 3
Explanation: From the top-left corner, there are a total of 3 ways to reach the bottom-right corner:
1. Right -> Down -> Down
2. Down -> Down -> Right
3. Down -> Right -> Down

 

Constraints:

    1 <= m, n <= 100

https://leetcode.com/problems/unique-paths/description/
"""

        def easy_solution(m: int, n: int) -> int:
            # Dynamic programming solution with O(m * n) space complexity
            dp = [[1] * n for _ in range(m)]
            for i in range(1, m):
                for j in range(1, n):
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
            return dp[m - 1][n - 1]

        def optimized_solution(m: int, n: int) -> int:
            # Optimized solution with O(n) space complexity
            row = [1] * n
            for _ in range(1, m):
                for j in range(1, n):
                    row[j] += row[j - 1]
            return row[-1]

        return Solution(
            question="Unique Paths",
            easy_solution=easy_solution,
            optimized_solution=optimized_solution,
            time_complexity="O(m * n)",
            space_complexity="O(n)",
            similar_questions=["Unique Paths II", "Minimum Path Sum", "Dungeon Game"],
            problem_statement=problem_statement,
        )

    @staticmethod
    def jump_game() -> Solution:
        """
        Generate a Solution object for the Jump Game problem.

        Returns:
            Solution: A Solution object containing details of the Jump Game problem.
        """

        def problem_statement() -> str:
            return """You are given an integer array nums. You are initially positioned at the array's first index, and each element in the array represents your maximum jump length at that position.

Return true if you can reach the last index, or false otherwise.

 

Example 1:

Input: nums = [2,3,1,1,4]
Output: true
Explanation: Jump 1 step from index 0 to 1, then 3 steps to the last index.

Example 2:

Input: nums = [3,2,1,0,4]
Output: false
Explanation: You will always arrive at index 3 no matter what. Its maximum jump length is 0, which makes it impossible to reach the last index.

 

Constraints:

    1 <= nums.length <= 104
    0 <= nums[i] <= 105

https://leetcode.com/problems/jump-game/description/
"""

        def easy_solution(nums: List[int]) -> bool:
            # Greedy solution with O(n) time complexity
            max_reach = 0
            for i, jump in enumerate(nums):
                if i > max_reach:
                    return False
                max_reach = max(max_reach, i + jump)
            return True

        def optimized_solution(nums: List[int]) -> bool:
            # Optimized greedy solution
            last_pos = len(nums) - 1
            for i in range(len(nums) - 1, -1, -1):
                if i + nums[i] >= last_pos:
                    last_pos = i
            return last_pos == 0

        return Solution(
            question="Jump Game",
            easy_solution=easy_solution,
            optimized_solution=optimized_solution,
            time_complexity="O(n)",
            space_complexity="O(1)",
            similar_questions=["Jump Game II"],
            problem_statement=problem_statement,
        )
