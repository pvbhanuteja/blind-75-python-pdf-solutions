"""
Module containing heap-related solutions for Blind 75 LeetCode questions.
"""

from typing import List
import heapq
from blind_75_solutions import Solution


class HeapSolutions:
    """Class containing heap-related solutions for Blind 75 LeetCode questions."""

    @staticmethod
    def find_median_from_data_stream() -> Solution:
        """
        Generate a Solution object for the Find Median from Data Stream problem.

        Returns:
            Solution: A Solution object containing details of the Find Median from Data Stream problem.
        """
        def problem_statement() -> str:
            return '''The MedianFinder class has two methods:

void addNum(int num) - Adds the integer num from the data stream to the data structure.
double findMedian() - Returns the median of all elements so far. Answers within 10-5 of the actual answer will be accepted.

Implement the MedianFinder class:
    MedianFinder() initializes the MedianFinder object.
    void addNum(int num) adds the integer num from the data stream to the data structure.
    double findMedian() returns the median of all elements so far. Answers within 10-5 of the actual answer will be accepted.

Example 1:

Input
["MedianFinder", "addNum", "addNum", "findMedian", "addNum", "findMedian"]
[[], [1], [2], [], [3], []]
Output
[null, null, null, 1.5, null, 2.0]

Explanation
MedianFinder medianFinder = new MedianFinder();
medianFinder.addNum(1);    // arr = [1]
medianFinder.addNum(2);    // arr = [1, 2]
medianFinder.findMedian(); // return 1.5 (i.e., (1 + 2) / 2)
medianFinder.addNum(3);    // arr[1, 2, 3]
medianFinder.findMedian(); // return 2.0

Constraints:

    -105 <= num <= 105
    There will be at least one element in the data structure before calling findMedian.
    At most 5 * 104 calls will be made to addNum and findMedian.
    
https://leetcode.com/problems/find-median-from-data-stream/description/
'''

        class MedianFinder:
            def __init__(self):
                # Two heaps: max heap for the lower half and min heap for the upper half
                self.small = []  # max heap
                self.large = []  # min heap

            def addNum(self, num: int) -> None:
                # Maintain balance between heaps
                if len(self.small) == len(self.large):
                    heapq.heappush(self.large, -heapq.heappushpop(self.small, -num))
                else:
                    heapq.heappush(self.small, -heapq.heappushpop(self.large, num))

            def findMedian(self) -> float:
                # Calculate median based on the number of elements
                if len(self.small) == len(self.large):
                    return float(self.large[0] - self.small[0]) / 2.0
                else:
                    return float(self.large[0])

        def easy_solution() -> MedianFinder:
            return MedianFinder()

        def optimized_solution() -> MedianFinder:
            return MedianFinder()

        return Solution(
            question="Find Median from Data Stream",
            easy_solution=easy_solution,
            optimized_solution=optimized_solution,
            time_complexity="O(log n) for addNum, O(1) for findMedian",
            space_complexity="O(n)",
            similar_questions=["Sliding Window Median", "Find Median from Data Stream II"],
            problem_statement=problem_statement
        )

    @staticmethod
    def top_k_frequent_elements() -> Solution:
        """
        Generate a Solution object for the Top K Frequent Elements problem.

        Returns:
            Solution: A Solution object containing details of the Top K Frequent Elements problem.
        """
        def problem_statement() -> str:
            return '''Given an integer array nums and an integer k, return the k most frequent elements. You may return the answer in any order.

Example 1:

Input: nums = [1,1,1,2,2,3], k = 2
Output: [1,2]

Example 2:

Input: nums = [1], k = 1
Output: [1]

Constraints:

    1 <= nums.length <= 105
    -104 <= nums[i] <= 104
    k is in the range [1, the number of unique elements in the array].
    It is guaranteed that the answer is unique.

https://leetcode.com/problems/top-k-frequent-elements/description/
'''

        def easy_solution(nums: List[int], k: int) -> List[int]:
            # Brute-force solution: Use a heap to find the k most frequent elements
            count = {}
            for num in nums:
                count[num] = count.get(num, 0) + 1
            
            return heapq.nlargest(k, count.keys(), key=count.get)

        def optimized_solution(nums: List[int], k: int) -> List[int]:
            # Optimized solution: Use bucket sort to find the k most frequent elements
            count = {}
            freq = [[] for _ in range(len(nums) + 1)]
            
            for num in nums:
                count[num] = count.get(num, 0) + 1
            for num, c in count.items():
                freq[c].append(num)
            
            res = []
            for i in range(len(freq) - 1, 0, -1):
                for num in freq[i]:
                    res.append(num)
                    if len(res) == k:
                        return res

        return Solution(
            question="Top K Frequent Elements",
            easy_solution=easy_solution,
            optimized_solution=optimized_solution,
            time_complexity="O(n)",
            space_complexity="O(n)",
            similar_questions=["Top K Frequent Words", "Sort Characters By Frequency"],
            problem_statement=problem_statement
        )

    @staticmethod
    def merge_k_sorted_lists() -> Solution:
        """
        Generate a Solution object for the Merge k Sorted Lists problem.

        Returns:
            Solution: A Solution object containing details of the Merge k Sorted Lists problem.
        """
        def problem_statement() -> str:
            return '''You are given an array of k linked-lists lists, each linked-list is sorted in ascending order.

Merge all the linked-lists into one sorted linked-list and return it.

Example 1:

Input: lists = [[1,4,5],[1,3,4],[2,6]]
Output: [1,1,2,3,4,4,5,6]
Explanation: The linked-lists are:
[
  1->4->5,
  1->3->4,
  2->6
]
merging them into one sorted list:
1->1->2->3->4->4->5->6

Example 2:

Input: lists = []
Output: []

Example 3:

Input: lists = [[]]
Output: []

Constraints:

    k == lists.length
    0 <= k <= 104
    0 <= lists[i].length <= 500
    -104 <= lists[i][j] <= 104
    lists[i] is sorted in ascending order.
    The sum of lists[i].length will not exceed 104.

https://leetcode.com/problems/merge-k-sorted-lists/description/
'''

        class ListNode:
            def __init__(self, val: int = 0, next: 'ListNode' = None):
                self.val = val
                self.next = next

        def easy_solution(lists: List[ListNode]) -> ListNode:
            # Brute-force solution: Collect all nodes and sort them
            nodes = []
            head = point = ListNode(0)
            for l in lists:
                while l:
                    nodes.append(l.val)
                    l = l.next
            for x in sorted(nodes):
                point.next = ListNode(x)
                point = point.next
            return head.next

        def optimized_solution(lists: List[ListNode]) -> ListNode:
            # Optimized solution: Use a heap to merge lists
            heap = []
            for i, l in enumerate(lists):
                if l:
                    heapq.heappush(heap, (l.val, i, l))
            
            dummy = ListNode(0)
            curr = dummy
            while heap:
                val, i, node = heapq.heappop(heap)
                curr.next = ListNode(val)
                curr = curr.next
                if node.next:
                    heapq.heappush(heap, (node.next.val, i, node.next))
            
            return dummy.next

        return Solution(
            question="Merge k Sorted Lists",
            easy_solution=easy_solution,
            optimized_solution=optimized_solution,
            time_complexity="O(N log k)",
            space_complexity="O(k)",
            similar_questions=["Merge Two Sorted Lists", "Ugly Number II"],
            problem_statement=problem_statement
        )