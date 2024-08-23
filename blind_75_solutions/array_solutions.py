## blind_75_solutions/array_solutions.py

"""
Module containing array-related solutions for Blind 75 LeetCode questions.
"""

from typing import List, Tuple
from blind_75_solutions import Solution


class ArraySolutions:
    """Class containing array-related solutions for Blind 75 LeetCode questions."""

    @staticmethod
    def two_sum() -> Solution:
        """
        Generate a Solution object for the Two Sum problem.

        Returns:
            Solution: A Solution object containing details of the Two Sum problem.
        """

        def problem_statement() -> str:
            return '''Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

You can return the answer in any order.

 

Example 1:

Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].

Example 2:

Input: nums = [3,2,4], target = 6
Output: [1,2]

Example 3:

Input: nums = [3,3], target = 6
Output: [0,1]

 

Constraints:

    2 <= nums.length <= 104
    -109 <= nums[i] <= 109
    -109 <= target <= 109
    Only one valid answer exists.

https://leetcode.com/problems/two-sum/description/
'''

        def easy_solution(nums: List[int], target: int) -> List[int]:
            # Brute-force solution: Check every pair of numbers
            for i in range(len(nums)):
                for j in range(i + 1, len(nums)):
                    if nums[i] + nums[j] == target:
                        return [i, j]
            return []

        def optimized_solution(nums: List[int], target: int) -> List[int]:
            # Optimized solution: Use a dictionary to track numbers and their indices
            num_dict = {}
            for i, num in enumerate(nums):
                complement = target - num
                if complement in num_dict:
                    return [num_dict[complement], i]
                num_dict[num] = i
            return []

        return Solution(
            question="Two Sum",
            easy_solution=easy_solution,
            optimized_solution=optimized_solution,
            time_complexity="O(n)",
            space_complexity="O(n)",
            similar_questions=["3Sum", "4Sum", "Two Sum II - Input array is sorted"],
            problem_statement=problem_statement
        )

    @staticmethod
    def best_time_to_buy_sell_stock() -> Solution:
        """
        Generate a Solution object for the Best Time to Buy and Sell Stock problem.

        Returns:
            Solution: A Solution object containing details of the Best Time to Buy and Sell Stock problem.
        """

        def problem_statement() -> str:
            return '''You are given an array prices where prices[i] is the price of a given stock on the ith day.

You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock.

Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return 0.

 

Example 1:

Input: prices = [7,1,5,3,6,4]
Output: 5
Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.
Note that buying on day 2 and selling on day 1 is not allowed because you must buy before you sell.

Example 2:

Input: prices = [7,6,4,3,1]
Output: 0
Explanation: In this case, no transactions are done and the max profit = 0.

 

Constraints:

    1 <= prices.length <= 105
    0 <= prices[i] <= 104

https://leetcode.com/problems/best-time-to-buy-and-sell-stock/description/
'''

        def easy_solution(prices: List[int]) -> int:
            # Brute-force solution: Check every pair of days
            max_profit = 0
            for i in range(len(prices)):
                for j in range(i + 1, len(prices)):
                    profit = prices[j] - prices[i]
                    if profit > max_profit:
                        max_profit = profit
            return max_profit

        def optimized_solution(prices: List[int]) -> int:
            # Optimized solution: Track the minimum price and maximum profit
            if not prices:
                return 0
            max_profit = 0
            min_price = float('inf')
            for price in prices:
                if price < min_price:
                    min_price = price
                else:
                    max_profit = max(max_profit, price - min_price)
            return max_profit

        return Solution(
            question="Best Time to Buy and Sell Stock",
            easy_solution=easy_solution,
            optimized_solution=optimized_solution,
            time_complexity="O(n)",
            space_complexity="O(1)",
            similar_questions=["Best Time to Buy and Sell Stock II", "Best Time to Buy and Sell Stock III"],
            problem_statement=problem_statement
        )

    @staticmethod
    def contains_duplicate() -> Solution:
        """
        Generate a Solution object for the Contains Duplicate problem.

        Returns:
            Solution: A Solution object containing details of the Contains Duplicate problem.
        """

        def problem_statement() -> str:
            return '''Given an integer array nums, return true if any value appears at least twice in the array, and return false if every element is distinct.
Given an integer array nums, return true if any value appears at least twice in the array, and return false if every element is distinct.

 

Example 1:

Input: nums = [1,2,3,1]
Output: true

Example 2:

Input: nums = [1,2,3,4]
Output: false

Example 3:

Input: nums = [1,1,1,3,3,4,3,2,4,2]
Output: true

 

Constraints:

    1 <= nums.length <= 105
    -109 <= nums[i] <= 109
    
https://leetcode.com/problems/contains-duplicate/description/
'''
        def easy_solution(nums: List[int]) -> bool:
            # Brute-force solution: Check every pair of numbers
            for i in range(len(nums)):
                for j in range(i + 1, len(nums)):
                    if nums[i] == nums[j]:
                        return True
            return False

        def optimized_solution(nums: List[int]) -> bool:
            # Optimized solution: Use a set to track seen numbers
            return len(nums) != len(set(nums))

        return Solution(
            question="Contains Duplicate",
            easy_solution=easy_solution,
            optimized_solution=optimized_solution,
            time_complexity="O(n)",
            space_complexity="O(n)",
            similar_questions=["Contains Duplicate II", "Contains Duplicate III"],
            problem_statement=problem_statement
        )

    @staticmethod
    def product_of_array_except_self() -> Solution:
        """
        Generate a Solution object for the Product of Array Except Self problem.

        Returns:
            Solution: A Solution object containing details of the Product of Array Except Self problem.
        """
        def problem_statement() -> str:
            return '''Given an integer array nums, return an array answer such that answer[i] is equal to the product of all the elements of nums except nums[i].

The product of any prefix or suffix of nums is guaranteed to fit in a 32-bit integer.

You must write an algorithm that runs in O(n) time and without using the division operation.

 

Example 1:

Input: nums = [1,2,3,4]
Output: [24,12,8,6]

Example 2:

Input: nums = [-1,1,0,-3,3]
Output: [0,0,9,0,0]

 

Constraints:

    2 <= nums.length <= 105
    -30 <= nums[i] <= 30
    The product of any prefix or suffix of nums is guaranteed to fit in a 32-bit integer.

 

Follow up: Can you solve the problem in O(1) extra space complexity? (The output array does not count as extra space for space complexity analysis.)

https://leetcode.com/problems/product-of-array-except-self/description/
'''

        def easy_solution(nums: List[int]) -> List[int]:
            # Brute-force solution: Calculate product for each index
            n = len(nums)
            result = [1] * n
            for i in range(n):
                for j in range(n):
                    if i != j:
                        result[i] *= nums[j]
            return result

        def optimized_solution(nums: List[int]) -> List[int]:
            # Optimized solution: Use left and right product arrays
            n = len(nums)
            result = [1] * n
            left_product = 1
            right_product = 1
            for i in range(n):
                result[i] *= left_product
                left_product *= nums[i]
                result[n - 1 - i] *= right_product
                right_product *= nums[n - 1 - i]
            return result

        return Solution(
            question="Product of Array Except Self",
            easy_solution=easy_solution,
            optimized_solution=optimized_solution,
            time_complexity="O(n)",
            space_complexity="O(1)",
            similar_questions=["Trapping Rain Water", "Maximum Product Subarray"],
            problem_statement=problem_statement
        )

    @staticmethod
    def maximum_subarray() -> Solution:
        """
        Generate a Solution object for the Maximum Subarray problem.

        Returns:
            Solution: A Solution object containing details of the Maximum Subarray problem.
        """
        def problem_statement() -> str:
            return '''Given an integer array nums, find the subarray with the largest sum, and return its sum.

A subarray is a contiguous part of an array.

 

Example 1:

Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
Output: 6
Explanation: The subarray [4,-1,2,1] has the largest sum 6.

Example 2:

Input: nums = [1]
Output: 1

Example 3:

Input: nums = [5,4,-1,7,8]
Output: 23

 

Constraints:

    1 <= nums.length <= 105
    -104 <= nums[i] <= 104

https://leetcode.com/problems/maximum-subarray/description/
'''

        def easy_solution(nums: List[int]) -> int:
            # Brute-force solution: Check all subarrays
            max_sum = float('-inf')
            for i in range(len(nums)):
                current_sum = 0
                for j in range(i, len(nums)):
                    current_sum += nums[j]
                    max_sum = max(max_sum, current_sum)
            return max_sum

        def optimized_solution(nums: List[int]) -> int:
            # Optimized solution: Use Kadane's algorithm
            max_sum = current_sum = nums[0]
            for num in nums[1:]:
                current_sum = max(num, current_sum + num)
                max_sum = max(max_sum, current_sum)
            return max_sum

        return Solution(
            question="Maximum Subarray",
            easy_solution=easy_solution,
            optimized_solution=optimized_solution,
            time_complexity="O(n)",
            space_complexity="O(1)",
            similar_questions=["Best Time to Buy and Sell Stock", "Maximum Product Subarray"],
            problem_statement=problem_statement
        )

    @staticmethod
    def maximum_product_subarray() -> Solution:
        """
        Generate a Solution object for the Maximum Product Subarray problem.

        Returns:
            Solution: A Solution object containing details of the Maximum Product Subarray problem.
        """
        def problem_statement() -> str:
            return '''Given an integer array nums, find the contiguous subarray within an array (containing at least one number) which has the largest product.

 

Example 1:

Input: nums = [2,3,-2,4]
Output: 6
Explanation: [2,3] has the largest product 6.

Example 2:

Input: nums = [-2,0,-1]
Output: 0
Explanation: The result cannot be 2, because [-2,-1] is not a subarray.

 

Constraints:

    1 <= nums.length <= 105
    -10 <= nums[i] <= 10
    The product of any prefix or suffix of nums is guaranteed to fit in a 32-bit integer.

https://leetcode.com/problems/maximum-product-subarray/description/
'''

        def easy_solution(nums: List[int]) -> int:
            # Brute-force solution: Check all subarrays
            max_product = float('-inf')
            for i in range(len(nums)):
                current_product = 1
                for j in range(i, len(nums)):
                    current_product *= nums[j]
                    max_product = max(max_product, current_product)
            return max_product

        def optimized_solution(nums: List[int]) -> int:
            # Optimized solution: Track max and min products
            max_so_far = min_so_far = result = nums[0]
            for i in range(1, len(nums)):
                temp_max = max(nums[i], max_so_far * nums[i], min_so_far * nums[i])
                min_so_far = min(nums[i], max_so_far * nums[i], min_so_far * nums[i])
                max_so_far = temp_max
                result = max(result, max_so_far)
            return result

        return Solution(
            question="Maximum Product Subarray",
            easy_solution=easy_solution,
            optimized_solution=optimized_solution,
            time_complexity="O(n)",
            space_complexity="O(1)",
            similar_questions=["Maximum Subarray", "House Robber"],
            problem_statement=problem_statement
        )

    @staticmethod
    def find_minimum_in_rotated_sorted_array() -> Solution:
        """
        Generate a Solution object for the Find Minimum in Rotated Sorted Array problem.

        Returns:
            Solution: A Solution object containing details of the Find Minimum in Rotated Sorted Array problem.
        """
        def problem_statement() -> str:
            return '''Suppose an array of length n sorted in ascending order is rotated between 1 and n times. For example, the array nums = [0,1,2,4,5,6,7] might become:

[4,5,6,7,0,1,2] if it was rotated 4 times.
[0,1,2,4,5,6,7] if it was rotated 7 times.

Notice that rotating an array [a[0], a[1], a[2], ..., a[n-1]] 1 time results in the array [a[n-1], a[0], a[1], a[2], ..., a[n-2]].

Given the sorted rotated array nums of unique elements, return the minimum element of this array.

You must write an algorithm that runs in O(log n) time.

 

Example 1:

Input: nums = [3,4,5,1,2]
Output: 1
Explanation: The original array was [1,2,3,4,5] rotated 3 times.

Example 2:

Input: nums = [4,5,6,7,0,1,2]
Output: 0
Explanation: The original array was [0,1,2,4,5,6,7] and it was rotated 4 times.

Example 3:

Input: nums = [11,13,15,17]
Output: 11
Explanation: The original array was [11,13,15,17] and it was rotated 4 times.

 

Constraints:

    n == nums.length
    1 <= n <= 5000
    -5000 <= nums[i] <= 5000
    All the integers of nums are unique.
    nums is guaranteed to be rotated at some pivot.

https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/description/
'''

        def easy_solution(nums: List[int]) -> int:
            # Brute-force solution: Use built-in min function
            return min(nums)

        def optimized_solution(nums: List[int]) -> int:
            # Optimized solution: Use binary search
            left, right = 0, len(nums) - 1
            while left < right:
                mid = left + (right - left) // 2
                if nums[mid] > nums[right]:
                    left = mid + 1
                else:
                    right = mid
            return nums[left]

        return Solution(
            question="Find Minimum in Rotated Sorted Array",
            easy_solution=easy_solution,
            optimized_solution=optimized_solution,
            time_complexity="O(log n)",
            space_complexity="O(1)",
            similar_questions=["Search in Rotated Sorted Array", "Find Minimum in Rotated Sorted Array II"],
            problem_statement=problem_statement
        )

    @staticmethod
    def search_in_rotated_sorted_array() -> Solution:
        """
        Generate a Solution object for the Search in Rotated Sorted Array problem.

        Returns:
            Solution: A Solution object containing details of the Search in Rotated Sorted Array problem.
        """
        def problem_statement() -> str:
            return '''There is an integer array nums sorted in ascending order (with distinct values).

Prior to being passed to your function, nums is possibly rotated at an unknown pivot index k (1 <= k < nums.length) such that the resulting array is [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]] (0-indexed). For example, [0,1,2,4,5,6,7] might become [4,5,6,7,0,1,2].

Given the array nums after the possible rotation and an integer target, return the index of target if it is in nums, or -1 if it is not in nums.

You must write an algorithm with O(log n) runtime complexity.

 

Example 1:

Input: nums = [4,5,6,7,0,1,2], target = 0
Output: 4

Example 2:

Input: nums = [4,5,6,7,0,1,2], target = 3
Output: -1

Example 3:

Input: nums = [1], target = 0
Output: -1

 

Constraints:

    1 <= nums.length <= 5000
    -104 <= nums[i] <= 104
    All values of nums are unique.
    nums is an ascending array that is possibly rotated.
    -104 <= target <= 104

https://leetcode.com/problems/search-in-rotated-sorted-array/description/
'''

        def easy_solution(nums: List[int], target: int) -> int:
            # Brute-force solution: Use built-in index function
            return nums.index(target) if target in nums else -1

        def optimized_solution(nums: List[int], target: int) -> int:
            # Optimized solution: Use binary search
            left, right = 0, len(nums) - 1
            while left <= right:
                mid = left + (right - left) // 2
                if nums[mid] == target:
                    return mid
                if nums[left] <= nums[mid]:
                    if nums[left] <= target < nums[mid]:
                        right = mid - 1
                    else:
                        left = mid + 1
                else:
                    if nums[mid] < target <= nums[right]:
                        left = mid + 1
                    else:
                        right = mid - 1
            return -1

        return Solution(
            question="Search in Rotated Sorted Array",
            easy_solution=easy_solution,
            optimized_solution=optimized_solution,
            time_complexity="O(log n)",
            space_complexity="O(1)",
            similar_questions=["Search in Rotated Sorted Array II", "Find Minimum in Rotated Sorted Array"],
            problem_statement=problem_statement
        )

    @staticmethod
    def three_sum() -> Solution:
        """
        Generate a Solution object for the 3Sum problem.

        Returns:
            Solution: A Solution object containing details of the 3Sum problem.
        """
        def problem_statement() -> str:
            return '''Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.

Notice that the solution set must not contain duplicate triplets.

 

Example 1:

Input: nums = [-1,0,1,2,-1,-4]
Output: [[-1,-1,2],[-1,0,1]]

Example 2:

Input: nums = []
Output: []

Example 3:

Input: nums = [0]
Output: []

 

Constraints:

    0 <= nums.length <= 3000
    -105 <= nums[i] <= 105

https://leetcode.com/problems/3sum/description/
'''

        def easy_solution(nums: List[int]) -> List[List[int]]:
            # Brute-force solution: Check all triplets
            result = []
            nums.sort()
            for i in range(len(nums) - 2):
                if i > 0 and nums[i] == nums[i - 1]:
                    continue
                for j in range(i + 1, len(nums) - 1):
                    if j > i + 1 and nums[j] == nums[j - 1]:
                        continue
                    for k in range(j + 1, len(nums)):
                        if k > j + 1 and nums[k] == nums[k - 1]:
                            continue
                        if nums[i] + nums[j] + nums[k] == 0:
                            result.append([nums[i], nums[j], nums[k]])
            return result

        def optimized_solution(nums: List[int]) -> List[List[int]]:
            # Optimized solution: Use two pointers
            result = []
            nums.sort()
            for i in range(len(nums) - 2):
                if i > 0 and nums[i] == nums[i - 1]:
                    continue
                left, right = i + 1, len(nums) - 1
                while left < right:
                    total = nums[i] + nums[left] + nums[right]
                    if total == 0:
                        result.append([nums[i], nums[left], nums[right]])
                        while left < right and nums[left] == nums[left + 1]:
                            left += 1
                        while left < right and nums[right] == nums[right - 1]:
                            right -= 1
                        left += 1
                        right -= 1
                    elif total < 0:
                        left += 1
                    else:
                        right -= 1
            return result

        return Solution(
            question="3Sum",
            easy_solution=easy_solution,
            optimized_solution=optimized_solution,
            time_complexity="O(n^2)",
            space_complexity="O(1)",
            similar_questions=["Two Sum", "3Sum Closest", "4Sum"],
            problem_statement=problem_statement
        )

    @staticmethod
    def container_with_most_water() -> Solution:
        """
        Generate a Solution object for the Container With Most Water problem.

        Returns:
            Solution: A Solution object containing details of the Container With Most Water problem.
        """
        def problem_statement() -> str:
            return '''You are given an integer array height of length n. There are n vertical lines drawn such that the two endpoints of the ith line are (i, 0) and (i, height[i]).

Find two lines that together with the x-axis form a container, such that the container contains the most water.

Return the maximum amount of water a container can store.

Notice that you may not slant the container.

 

Example 1:

Input: height = [1,8,6,2,5,4,8,3,7]
Output: 49
Explanation: The above vertical lines are represented by array [1,8,6,2,5,4,8,3,7]. In this case, the max area of water (blue section) the container can contain is 49.

Example 2:

Input: height = [1,1]
Output: 1

 

Constraints:

    n == height.length
    2 <= n <= 105
    0 <= height[i] <= 104

https://leetcode.com/problems/container-with-most-water/description/
'''

        def easy_solution(height: List[int]) -> int:
            # Brute-force solution: Check all pairs of lines
            max_area = 0
            for i in range(len(height)):
                for j in range(i + 1, len(height)):
                    area = min(height[i], height[j]) * (j - i)
                    max_area = max(max_area, area)
            return max_area

        def optimized_solution(height: List[int]) -> int:
            # Optimized solution: Use two pointers
            max_area = 0
            left, right = 0, len(height) - 1
            while left < right:
                area = min(height[left], height[right]) * (right - left)
                max_area = max(max_area, area)
                if height[left] < height[right]:
                    left += 1
                else:
                    right -= 1
            return max_area

        return Solution(
            question="Container With Most Water",
            easy_solution=easy_solution,
            optimized_solution=optimized_solution,
            time_complexity="O(n)",
            space_complexity="O(1)",
            similar_questions=["Trapping Rain Water"],
            problem_statement=problem_statement
        )