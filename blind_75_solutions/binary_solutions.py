## blind_75_solutions/binary_solutions.py

"""
Module containing binary-related solutions for Blind 75 LeetCode questions.
"""

from typing import List
from blind_75_solutions import Solution


class BinarySolutions:
    """Class containing binary-related solutions for Blind 75 LeetCode questions."""

    @staticmethod
    def sum_of_two_integers() -> Solution:
        """
        Generate a Solution object for the Sum of Two Integers problem.

        Returns:
            Solution: A Solution object containing details of the Sum of Two Integers problem.
        """
        def problem_statement() -> str:
            return '''Given two integers a and b, return the sum of the two integers without using the operators + and -.

Example 1:

Input: a = 1, b = 2
Output: 3

Example 2:

Input: a = 2, b = 3
Output: 5

Constraints:

    -1000 <= a, b <= 1000

https://leetcode.com/problems/sum-of-two-integers/
'''

        def easy_solution(a: int, b: int) -> int:
            # Iterative solution: Use bit manipulation to add without + or -
            while b != 0:
                carry = a & b
                a = a ^ b
                b = carry << 1
            return a

        def optimized_solution(a: int, b: int) -> int:
            # The easy solution is already optimal for this problem
            return easy_solution(a, b)

        return Solution(
            question="Sum of Two Integers",
            easy_solution=easy_solution,
            optimized_solution=optimized_solution,
            time_complexity="O(1)",
            space_complexity="O(1)",
            similar_questions=["Add Binary"],
            problem_statement=problem_statement
        )

    @staticmethod
    def number_of_1_bits() -> Solution:
        """
        Generate a Solution object for the Number of 1 Bits problem.

        Returns:
            Solution: A Solution object containing details of the Number of 1 Bits problem.
        """
        def problem_statement() -> str:
            return '''Write a function that takes an unsigned integer and returns the number of '1' bits it has (also known as the Hamming weight).

Example 1:

Input: n = 00000000000000000000000000001011
Output: 3
Explanation: The input binary string 00000000000000000000000000001011 has a total of three '1' bits.

Example 2:

Input: n = 00000000000000000000000010000000
Output: 1
Explanation: The input binary string 00000000000000000000000010000000 has a total of one '1' bit.

Example 3:

Input: n = 11111111111111111111111111111101
Output: 31
Explanation: The input binary string 11111111111111111111111111111101 has a total of thirty-one '1' bits.

Constraints:

    The input must be a binary string of length 32.

https://leetcode.com/problems/number-of-1-bits/
'''

        def easy_solution(n: int) -> int:
            # Iterative solution: Count bits by shifting
            count = 0
            while n:
                count += n & 1
                n >>= 1
            return count

        def optimized_solution(n: int) -> int:
            # Optimized solution: Use n & (n-1) to count bits
            count = 0
            while n:
                n &= n - 1
                count += 1
            return count

        return Solution(
            question="Number of 1 Bits",
            easy_solution=easy_solution,
            optimized_solution=optimized_solution,
            time_complexity="O(1)",
            space_complexity="O(1)",
            similar_questions=["Hamming Distance", "Reverse Bits"],
            problem_statement=problem_statement
        )

    @staticmethod
    def counting_bits() -> Solution:
        """
        Generate a Solution object for the Counting Bits problem.

        Returns:
            Solution: A Solution object containing details of the Counting Bits problem.
        """
        def problem_statement() -> str:
            return '''Given an integer n, return an array ans of length n + 1 such that for each i (0 <= i <= n), ans[i] is the number of 1's in the binary representation of i.

Example 1:

Input: n = 2
Output: [0,1,1]
Explanation:
0 --> 0
1 --> 1
2 --> 10

Example 2:

Input: n = 5
Output: [0,1,1,2,1,2]
Explanation:
0 --> 0
1 --> 1
2 --> 10
3 --> 11
4 --> 100
5 --> 101

Constraints:

    0 <= n <= 105

https://leetcode.com/problems/counting-bits/
'''

        def easy_solution(n: int) -> List[int]:
            # Iterative solution: Count bits for each number
            def count_bits(x: int) -> int:
                count = 0
                while x:
                    count += x & 1
                    x >>= 1
                return count
            
            return [count_bits(i) for i in range(n + 1)]

        def optimized_solution(n: int) -> List[int]:
            # Optimized solution: Use dynamic programming
            dp = [0] * (n + 1)
            for i in range(1, n + 1):
                dp[i] = dp[i >> 1] + (i & 1)
            return dp

        return Solution(
            question="Counting Bits",
            easy_solution=easy_solution,
            optimized_solution=optimized_solution,
            time_complexity="O(n)",
            space_complexity="O(n)",
            similar_questions=["Number of 1 Bits", "Binary Watch"],
            problem_statement=problem_statement
        )

    @staticmethod
    def missing_number() -> Solution:
        """
        Generate a Solution object for the Missing Number problem.

        Returns:
            Solution: A Solution object containing details of the Missing Number problem.
        """
        def problem_statement() -> str:
            return '''Given an array nums containing n distinct numbers in the range [0, n], return the only number in the range that is missing from the array.

Example 1:

Input: nums = [3,0,1]
Output: 2
Explanation: n = 3 since there are 3 numbers, so all numbers are in the range [0,3]. 2 is the missing number in the range since it does not appear in nums.

Example 2:

Input: nums = [0,1]
Output: 2
Explanation: n = 2 since there are 2 numbers, so all numbers are in the range [0,2]. 2 is the missing number in the range since it does not appear in nums.

Example 3:

Input: nums = [9,6,4,2,3,5,7,0,1]
Output: 8
Explanation: n = 9 since there are 9 numbers, so all numbers are in the range [0,9]. 8 is the missing number in the range since it does not appear in nums.

Constraints:

    n == nums.length
    1 <= n <= 104
    0 <= nums[i] <= n
    All the numbers of nums are unique.

https://leetcode.com/problems/missing-number/
'''

        def easy_solution(nums: List[int]) -> int:
            # Iterative solution: Use summation formula
            n = len(nums)
            total_sum = n * (n + 1) // 2
            return total_sum - sum(nums)

        def optimized_solution(nums: List[int]) -> int:
            # Optimized solution: Use XOR
            missing = len(nums)
            for i, num in enumerate(nums):
                missing ^= i ^ num
            return missing

        return Solution(
            question="Missing Number",
            easy_solution=easy_solution,
            optimized_solution=optimized_solution,
            time_complexity="O(n)",
            space_complexity="O(1)",
            similar_questions=["Find the Duplicate Number", "Find All Numbers Disappeared in an Array"],
            problem_statement=problem_statement
        )

    @staticmethod
    def reverse_bits() -> Solution:
        """
        Generate a Solution object for the Reverse Bits problem.

        Returns:
            Solution: A Solution object containing details of the Reverse Bits problem.
        """
        def problem_statement() -> str:
            return '''Reverse bits of a given 32 bits unsigned integer.

Note:

    Note that in some languages, such as Java, there is no unsigned integer type. In this case, both input and output will be given as a signed integer type. They should not affect your implementation, as the integer's internal binary representation is the same, whether it is signed or unsigned.
    In Java, the compiler represents the signed integers using 2's complement notation. Therefore, in Example 2 above, the input represents the signed integer -3 and the output represents the signed integer -1073741825.

Example 1:

Input: n = 00000010100101000001111010011100
Output: 964176192 (00111001011110000010100101000000)
Explanation: The input binary string 00000010100101000001111010011100 represents the unsigned integer 43261596, so return 964176192 which its binary representation is 00111001011110000010100101000000.

Example 2:

Input: n = 11111111111111111111111111111101
Output: 3221225471 (10111111111111111111111111111111)
Explanation: The input binary string 11111111111111111111111111111101 represents the unsigned integer 4294967293, so return 3221225471 which its binary representation is 10111111111111111111111111111111.

Constraints:

    The input must be a binary string of length 32

https://leetcode.com/problems/reverse-bits/
'''

        def easy_solution(n: int) -> int:
            # Iterative solution: Reverse bits by shifting
            result = 0
            for i in range(32):
                result <<= 1
                result |= n & 1
                n >>= 1
            return result

        def optimized_solution(n: int) -> int:
            # The easy solution is already optimal for this problem
            return easy_solution(n)

        return Solution(
            question="Reverse Bits",
            easy_solution=easy_solution,
            optimized_solution=optimized_solution,
            time_complexity="O(1)",
            space_complexity="O(1)",
            similar_questions=["Number of 1 Bits"],
            problem_statement=problem_statement
        )