"""
Module containing interval-related solutions for Blind 75 LeetCode questions.
"""

from typing import List
from blind_75_solutions import Solution


class IntervalSolutions:
    """Class containing interval-related solutions for Blind 75 LeetCode questions."""

    @staticmethod
    def insert_interval() -> Solution:
        """
        Generate a Solution object for the Insert Interval problem.

        Returns:
            Solution: A Solution object containing details of the Insert Interval problem.
        """
        def problem_statement() -> str:
            return '''You are given an array of non-overlapping intervals intervals where intervals[i] = [starti, endi] represent the start and the end of the ith interval and intervals is sorted in ascending order by starti. You are also given an interval newInterval = [start, end] that represents the start and end of another interval.

Insert newInterval into intervals such that intervals is still sorted in ascending order by starti and intervals still does not have any overlapping intervals (merge overlapping intervals if necessary).

Return intervals after the insertion.

Example 1:

Input: intervals = [[1,3],[6,9]], newInterval = [2,5]
Output: [[1,5],[6,9]]

Example 2:

Input: intervals = [[1,2],[3,5],[6,7],[8,10],[12,16]], newInterval = [4,8]
Output: [[1,2],[3,10],[12,16]]
Explanation: Because the new interval [4,8] overlaps with [3,5],[6,7],[8,10].

Constraints:

    0 <= intervals.length <= 104
    intervals[i].length == 2
    0 <= starti <= endi <= 105
    intervals is sorted by starti in ascending order.
    newInterval.length == 2
    0 <= start <= end <= 105

https://leetcode.com/problems/insert-interval/description/
'''

        def easy_solution(intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
            # Easy solution: Iterate through intervals, merge if overlapping, otherwise append directly
            result = []
            i = 0
            n = len(intervals)

            # Add intervals before newInterval
            while i < n and intervals[i][1] < newInterval[0]:
                result.append(intervals[i])
                i += 1

            # Merge overlapping intervals
            while i < n and intervals[i][0] <= newInterval[1]:
                newInterval[0] = min(newInterval[0], intervals[i][0])
                newInterval[1] = max(newInterval[1], intervals[i][1])
                i += 1

            result.append(newInterval)

            # Add remaining intervals
            while i < n:
                result.append(intervals[i])
                i += 1

            return result

        def optimized_solution(intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
            # Optimized solution: Iterate through intervals and merge on the go
            result = []
            for interval in intervals:
                if interval[1] < newInterval[0]:
                    result.append(interval)
                elif interval[0] > newInterval[1]:
                    result.append(newInterval)
                    newInterval = interval
                else:
                    newInterval[0] = min(newInterval[0], interval[0])
                    newInterval[1] = max(newInterval[1], interval[1])
            result.append(newInterval)
            return result

        return Solution(
            question="Insert Interval",
            easy_solution=easy_solution,
            optimized_solution=optimized_solution,
            time_complexity="O(n)",
            space_complexity="O(n)",
            similar_questions=["Merge Intervals", "Range Module"],
            problem_statement=problem_statement
        )

    @staticmethod
    def merge_intervals() -> Solution:
        """
        Generate a Solution object for the Merge Intervals problem.

        Returns:
            Solution: A Solution object containing details of the Merge Intervals problem.
        """
        def problem_statement() -> str:
            return '''Given an array of intervals where intervals[i] = [starti, endi], merge all overlapping intervals, and return an array of the non-overlapping intervals that cover all the intervals in the input.

Example 1:

Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
Output: [[1,6],[8,10],[15,18]]
Explanation: Since intervals [1,3] and [2,6] overlaps, merge them into [1,6].

Example 2:

Input: intervals = [[1,4],[4,5]]
Output: [[1,5]]
Explanation: Intervals [1,4] and [4,5] are considered overlapping.

Constraints:

    1 <= intervals.length <= 104
    intervals[i].length == 2
    0 <= starti <= endi <= 104

https://leetcode.com/problems/merge-intervals/description/
'''

        def easy_solution(intervals: List[List[int]]) -> List[List[int]]:
            # Easy solution: Sort intervals and merge sequentially
            if not intervals:
                return []

            intervals.sort(key=lambda x: x[0])
            merged = [intervals[0]]

            for interval in intervals[1:]:
                if interval[0] <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], interval[1])
                else:
                    merged.append(interval)

            return merged

        def optimized_solution(intervals: List[List[int]]) -> List[List[int]]:
            # Optimized solution: Similar to easy solution but checks if merged list is empty
            if not intervals:
                return []

            intervals.sort(key=lambda x: x[0])
            merged = []

            for interval in intervals:
                if not merged or merged[-1][1] < interval[0]:
                    merged.append(interval)
                else:
                    merged[-1][1] = max(merged[-1][1], interval[1])

            return merged

        return Solution(
            question="Merge Intervals",
            easy_solution=easy_solution,
            optimized_solution=optimized_solution,
            time_complexity="O(n log n)",
            space_complexity="O(n)",
            similar_questions=["Insert Interval", "Meeting Rooms", "Meeting Rooms II", "Teemo Attacking", "Add Bold Tag in String", "Range Module", "Employee Free Time", "Partition Labels"],
            problem_statement=problem_statement
        )

    @staticmethod
    def non_overlapping_intervals() -> Solution:
        """
        Generate a Solution object for the Non-overlapping Intervals problem.

        Returns:
            Solution: A Solution object containing details of the Non-overlapping Intervals problem.
        """
        def problem_statement() -> str:
            return '''Given an array of intervals intervals where intervals[i] = [starti, endi], return the minimum number of intervals you need to remove to make the rest of the intervals non-overlapping.

Example 1:

Input: intervals = [[1,2],[2,3],[3,4],[1,3]]
Output: 1
Explanation: [1,3] can be removed and the rest of the intervals are non-overlapping.

Example 2:

Input: intervals = [[1,2],[1,2],[1,2]]
Output: 2
Explanation: You need to remove two [1,2] to make the rest of the intervals non-overlapping.

Example 3:

Input: intervals = [[1,2],[2,3]]
Output: 0
Explanation: You don't need to remove any of the intervals since they're already non-overlapping.

Constraints:

    1 <= intervals.length <= 105
    intervals[i].length == 2
    -5 * 104 <= starti < endi <= 5 * 104

https://leetcode.com/problems/non-overlapping-intervals/description/
'''

        def easy_solution(intervals: List[List[int]]) -> int:
            # Easy solution: Sort by end time and count overlaps
            if not intervals:
                return 0

            intervals.sort(key=lambda x: x[1])
            count = 0
            end = float('-inf')

            for interval in intervals:
                if interval[0] >= end:
                    end = interval[1]
                else:
                    count += 1

            return count

        def optimized_solution(intervals: List[List[int]]) -> int:
            # Optimized solution: Similar to easy solution but starts with the first interval
            if not intervals:
                return 0

            intervals.sort(key=lambda x: x[1])
            count = 0
            end = intervals[0][1]

            for i in range(1, len(intervals)):
                if intervals[i][0] < end:
                    count += 1
                else:
                    end = intervals[i][1]

            return count

        return Solution(
            question="Non-overlapping Intervals",
            easy_solution=easy_solution,
            optimized_solution=optimized_solution,
            time_complexity="O(n log n)",
            space_complexity="O(1)",
            similar_questions=["Minimum Number of Arrows to Burst Balloons"],
            problem_statement=problem_statement
        )

    @staticmethod
    def meeting_rooms() -> Solution:
        """
        Generate a Solution object for the Meeting Rooms problem.

        Returns:
            Solution: A Solution object containing details of the Meeting Rooms problem.
        """
        def problem_statement() -> str:
            return '''Given an array of meeting time intervals where intervals[i] = [starti, endi], determine if a person could attend all meetings.

Example 1:

Input: intervals = [[0,30],[5,10],[15,20]]
Output: false

Example 2:

Input: intervals = [[7,10],[2,4]]
Output: true

Constraints:

    0 <= intervals.length <= 104
    intervals[i].length == 2
    0 <= starti < endi <= 106

https://leetcode.com/problems/meeting-rooms/description/
'''

        def easy_solution(intervals: List[List[int]]) -> bool:
            # Easy solution: Sort intervals by start time and check for overlaps
            intervals.sort(key=lambda x: x[0])

            for i in range(1, len(intervals)):
                if intervals[i][0] < intervals[i-1][1]:
                    return False

            return True

        def optimized_solution(intervals: List[List[int]]) -> bool:
            # Optimized solution: Sort start and end times separately and compare
            start_times = sorted(interval[0] for interval in intervals)
            end_times = sorted(interval[1] for interval in intervals)

            for i in range(1, len(intervals)):
                if start_times[i] < end_times[i-1]:
                    return False

            return True

        return Solution(
            question="Meeting Rooms",
            easy_solution=easy_solution,
            optimized_solution=optimized_solution,
            time_complexity="O(n log n)",
            space_complexity="O(n)",
            similar_questions=["Merge Intervals", "Meeting Rooms II"],
            problem_statement=problem_statement
        )

    @staticmethod
    def meeting_rooms_ii() -> Solution:
        """
        Generate a Solution object for the Meeting Rooms II problem.

        Returns:
            Solution: A Solution object containing details of the Meeting Rooms II problem.
        """
        def problem_statement() -> str:
            return '''Given an array of meeting time intervals intervals where intervals[i] = [starti, endi], return the minimum number of conference rooms required.

Example 1:

Input: intervals = [[0,30],[5,10],[15,20]]
Output: 2

Example 2:

Input: intervals = [[7,10],[2,4]]
Output: 1

Constraints:

    0 <= intervals.length <= 104
    intervals[i].length == 2
    0 <= starti < endi <= 106

https://leetcode.com/problems/meeting-rooms-ii/description/
'''

        def easy_solution(intervals: List[List[int]]) -> int:
            # Easy solution: Sort start and end times and use two pointers
            start_times = sorted(interval[0] for interval in intervals)
            end_times = sorted(interval[1] for interval in intervals)

            rooms = 0
            end_ptr = 0

            for start in start_times:
                if start < end_times[end_ptr]:
                    rooms += 1
                else:
                    end_ptr += 1

            return rooms

        def optimized_solution(intervals: List[List[int]]) -> int:
            # Optimized solution: Use a list of events and sort them
            events = []
            for start, end in intervals:
                events.append((start, 1))
                events.append((end, -1))

            events.sort(key=lambda x: (x[0], -x[1]))

            rooms = 0
            max_rooms = 0

            for _, event_type in events:
                rooms += event_type
                max_rooms = max(max_rooms, rooms)

            return max_rooms

        return Solution(
            question="Meeting Rooms II",
            easy_solution=easy_solution,
            optimized_solution=optimized_solution,
            time_complexity="O(n log n)",
            space_complexity="O(n)",
            similar_questions=["Merge Intervals", "Meeting Rooms", "Minimum Number of Arrows to Burst Balloons", "Car Pooling"],
            problem_statement=problem_statement
        )