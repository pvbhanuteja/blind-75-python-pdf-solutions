## blind_75_solutions/linked_list_solutions.py

"""
Module containing linked list-related solutions for Blind 75 LeetCode questions.
"""

from typing import Optional, List
from blind_75_solutions import Solution


class ListNode:
    """Definition for singly-linked list."""
    def __init__(self, val: int = 0, next: Optional['ListNode'] = None):
        self.val = val
        self.next = next


class LinkedListSolutions:
    """Class containing linked list-related solutions for Blind 75 LeetCode questions."""

    @staticmethod
    def reverse_linked_list() -> Solution:
        """
        Generate a Solution object for the Reverse Linked List problem.

        Returns:
            Solution: A Solution object containing details of the Reverse Linked List problem.
        """
        def problem_statement() -> str:
            return '''Given the head of a singly linked list, reverse the list, and return the reversed list.

Example 1:

Input: head = [1,2,3,4,5]
Output: [5,4,3,2,1]

Example 2:

Input: head = [1,2]
Output: [2,1]

Example 3:

Input: head = []
Output: []

Constraints:

    The number of nodes in the list is the range [0, 5000].
    -5000 <= Node.val <= 5000

https://leetcode.com/problems/reverse-linked-list/description/
'''

        def easy_solution(head: Optional[ListNode]) -> Optional[ListNode]:
            # Iterative approach: Reverse the list by changing the next pointers
            prev = None
            current = head
            while current:
                next_node = current.next
                current.next = prev
                prev = current
                current = next_node
            return prev

        def optimized_solution(head: Optional[ListNode]) -> Optional[ListNode]:
            # The easy solution is already optimal for this problem
            return easy_solution(head)

        return Solution(
            question="Reverse Linked List",
            easy_solution=easy_solution,
            optimized_solution=optimized_solution,
            time_complexity="O(n)",
            space_complexity="O(1)",
            similar_questions=["Reverse Linked List II", "Binary Tree Upside Down", "Palindrome Linked List"],
            problem_statement=problem_statement
        )

    @staticmethod
    def linked_list_cycle() -> Solution:
        """
        Generate a Solution object for the Linked List Cycle problem.

        Returns:
            Solution: A Solution object containing details of the Linked List Cycle problem.
        """
        def problem_statement() -> str:
            return '''Given head, the head of a linked list, determine if the linked list has a cycle in it.

There is a cycle in a linked list if there is some node in the list that can be reached again by continuously following the next pointer. Internally, pos is used to denote the index of the node that tail's next pointer is connected to. Note that pos is not passed as a parameter.

Return true if there is a cycle in the linked list. Otherwise, return false.

Example 1:

Input: head = [3,2,0,-4], pos = 1
Output: true
Explanation: There is a cycle in the linked list, where the tail connects to the 1st node (0-indexed).

Example 2:

Input: head = [1,2], pos = 0
Output: true
Explanation: There is a cycle in the linked list, where the tail connects to the 0th node.

Example 3:

Input: head = [1], pos = -1
Output: false
Explanation: There is no cycle in the linked list.

Constraints:

    The number of the nodes in the list is in the range [0, 104].
    -105 <= Node.val <= 105
    pos is -1 or a valid index in the linked list.

https://leetcode.com/problems/linked-list-cycle/description/
'''

        def easy_solution(head: Optional[ListNode]) -> bool:
            # Brute-force solution: Use a set to track visited nodes
            seen = set()
            current = head
            while current:
                if current in seen:
                    return True
                seen.add(current)
                current = current.next
            return False

        def optimized_solution(head: Optional[ListNode]) -> bool:
            # Optimized solution: Use two pointers (Floyd's Tortoise and Hare)
            if not head or not head.next:
                return False
            slow = head
            fast = head.next
            while slow != fast:
                if not fast or not fast.next:
                    return False
                slow = slow.next
                fast = fast.next.next
            return True

        return Solution(
            question="Linked List Cycle",
            easy_solution=easy_solution,
            optimized_solution=optimized_solution,
            time_complexity="O(n)",
            space_complexity="O(1)",
            similar_questions=["Linked List Cycle II", "Happy Number"],
            problem_statement=problem_statement
        )

    @staticmethod
    def merge_two_sorted_lists() -> Solution:
        """
        Generate a Solution object for the Merge Two Sorted Lists problem.

        Returns:
            Solution: A Solution object containing details of the Merge Two Sorted Lists problem.
        """
        def problem_statement() -> str:
            return '''You are given the heads of two sorted linked lists list1 and list2.

Merge the two lists in a one sorted list. The list should be made by splicing together the nodes of the first two lists.

Return the head of the merged linked list.

Example 1:

Input: list1 = [1,2,4], list2 = [1,3,4]
Output: [1,1,2,3,4,4]

Example 2:

Input: list1 = [], list2 = []
Output: []

Example 3:

Input: list1 = [], list2 = [0]
Output: [0]

Constraints:

    The number of nodes in both lists is in the range [0, 50].
    -100 <= Node.val <= 100
    Both list1 and list2 are sorted in non-decreasing order.

https://leetcode.com/problems/merge-two-sorted-lists/description/
'''

        def easy_solution(l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
            # Iterative approach: Merge two sorted lists
            dummy = ListNode(0)
            current = dummy
            while l1 and l2:
                if l1.val <= l2.val:
                    current.next = l1
                    l1 = l1.next
                else:
                    current.next = l2
                    l2 = l2.next
                current = current.next
            current.next = l1 if l1 else l2
            return dummy.next

        def optimized_solution(l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
            # The easy solution is already optimal for this problem
            return easy_solution(l1, l2)

        return Solution(
            question="Merge Two Sorted Lists",
            easy_solution=easy_solution,
            optimized_solution=optimized_solution,
            time_complexity="O(n + m)",
            space_complexity="O(1)",
            similar_questions=["Merge k Sorted Lists", "Merge Sorted Array", "Sort List", "Shortest Word Distance II"],
            problem_statement=problem_statement
        )

    @staticmethod
    def remove_nth_node_from_end_of_list() -> Solution:
        """
        Generate a Solution object for the Remove Nth Node From End of List problem.

        Returns:
            Solution: A Solution object containing details of the Remove Nth Node From End of List problem.
        """
        def problem_statement() -> str:
            return '''Given the head of a linked list, remove the nth node from the end of the list and return its head.

Example 1:

Input: head = [1,2,3,4,5], n = 2
Output: [1,2,3,5]

Example 2:

Input: head = [1], n = 1
Output: []

Example 3:

Input: head = [1,2], n = 1
Output: [1]

Constraints:

    The number of nodes in the list is sz.
    1 <= sz <= 30
    0 <= Node.val <= 100
    1 <= n <= sz

https://leetcode.com/problems/remove-nth-node-from-end-of-list/description/
'''

        def easy_solution(head: Optional[ListNode], n: int) -> Optional[ListNode]:
            # Two-pointer approach: Remove the nth node from the end
            dummy = ListNode(0)
            dummy.next = head
            first = dummy
            second = dummy
            for _ in range(n + 1):
                first = first.next
            while first:
                first = first.next
                second = second.next
            second.next = second.next.next
            return dummy.next

        def optimized_solution(head: Optional[ListNode], n: int) -> Optional[ListNode]:
            # The easy solution is already optimal for this problem
            return easy_solution(head, n)

        return Solution(
            question="Remove Nth Node From End of List",
            easy_solution=easy_solution,
            optimized_solution=optimized_solution,
            time_complexity="O(L)",
            space_complexity="O(1)",
            similar_questions=["Swapping Nodes in a Linked List", "Delete N Nodes After M Nodes of a Linked List"],
            problem_statement=problem_statement
        )

    @staticmethod
    def reorder_list() -> Solution:
        """
        Generate a Solution object for the Reorder List problem.

        Returns:
            Solution: A Solution object containing details of the Reorder List problem.
        """
        def problem_statement() -> str:
            return '''You are given the head of a singly linked-list. The list can be represented as:

L0 → L1 → … → Ln - 1 → Ln

Reorder the list to be on the following form:

L0 → Ln → L1 → Ln - 1 → L2 → Ln - 2 → …

You may not modify the values in the list's nodes. Only nodes themselves may be changed.

Example 1:

Input: head = [1,2,3,4]
Output: [1,4,2,3]

Example 2:

Input: head = [1,2,3,4,5]
Output: [1,5,2,4,3]

Constraints:

    The number of nodes in the list is in the range [1, 5 * 104].
    1 <= Node.val <= 1000

https://leetcode.com/problems/reorder-list/description/
'''

        def easy_solution(head: Optional[ListNode]) -> None:
            # Optimal approach: Find middle, reverse second half, and merge
            if not head or not head.next:
                return

            # Find the middle of the list
            slow = fast = head
            while fast.next and fast.next.next:
                slow = slow.next
                fast = fast.next.next

            # Reverse the second half of the list
            second = slow.next
            slow.next = None
            prev = None
            while second:
                next_node = second.next
                second.next = prev
                prev = second
                second = next_node

            # Merge the two halves
            first = head
            second = prev
            while second:
                next_first = first.next
                next_second = second.next
                first.next = second
                second.next = next_first
                first = next_first
                second = next_second

        def optimized_solution(head: Optional[ListNode]) -> None:
            # The easy solution is already optimal for this problem
            easy_solution(head)

        return Solution(
            question="Reorder List",
            easy_solution=easy_solution,
            optimized_solution=optimized_solution,
            time_complexity="O(n)",
            space_complexity="O(1)",
            similar_questions=["Palindrome Linked List"],
            problem_statement=problem_statement
        )

    @staticmethod
    def intersection_of_two_linked_lists() -> Solution:
        """
        Generate a Solution object for the Intersection of Two Linked Lists problem.

        Returns:
            Solution: A Solution object containing details of the Intersection of Two Linked Lists problem.
        """
        def problem_statement() -> str:
            return '''Given the heads of two singly linked-lists headA and headB, return the node at which the two lists intersect. If the two linked lists have no intersection at all, return null.

For example, the following two linked lists begin to intersect at node c1:

Example 1:

Input: intersectVal = 8, listA = [4,1,8,4,5], listB = [5,6,1,8,4,5], skipA = 2, skipB = 3
Output: Intersected at '8'
Explanation: The intersected node's value is 8 (note that this must not be 0 if the two lists intersect).
From the head of A, it reads as [4,1,8,4,5]. From the head of B, it reads as [5,6,1,8,4,5]. There are 2 nodes before the intersected node in A; There are 3 nodes before the intersected node in B.

Example 2:

Input: intersectVal = 2, listA = [1,9,1,2,4], listB = [3,2,4], skipA = 3, skipB = 1
Output: Intersected at '2'
Explanation: The intersected node's value is 2 (note that this must not be 0 if the two lists intersect).
From the head of A, it reads as [1,9,1,2,4]. From the head of B, it reads as [3,2,4]. There are 3 nodes before the intersected node in A; There are 1 node before the intersected node in B.

Example 3:

Input: intersectVal = 0, listA = [2,6,4], listB = [1,5], skipA = 3, skipB = 2
Output: No intersection
Explanation: From the head of A, it reads as [2,6,4]. From the head of B, it reads as [1,5]. Since the two lists do not intersect, intersectVal must be 0.

Constraints:

    The number of nodes of listA is in the m.
    The number of nodes of listB is in the n.
    1 <= m, n <= 3 * 104
    1 <= Node.val <= 105
    0 <= skipA < m
    0 <= skipB < n
    intersectVal is 0 if listA and listB do not intersect.
    intersectVal == listA[skipA] == listB[skipB] if listA and listB intersect.

https://leetcode.com/problems/intersection-of-two-linked-lists/description/
'''

        def easy_solution(headA: ListNode, headB: ListNode) -> Optional[ListNode]:
            # Two-pointer approach: Find intersection node
            if not headA or not headB:
                return None

            nodeA = headA
            nodeB = headB

            while nodeA != nodeB:
                nodeA = nodeA.next if nodeA else headB
                nodeB = nodeB.next if nodeB else headA

            return nodeA

        def optimized_solution(headA: ListNode, headB: ListNode) -> Optional[ListNode]:
            # The easy solution is already optimal for this problem
            return easy_solution(headA, headB)

        return Solution(
            question="Intersection of Two Linked Lists",
            easy_solution=easy_solution,
            optimized_solution=optimized_solution,
            time_complexity="O(n + m)",
            space_complexity="O(1)",
            similar_questions=["Minimum Index Sum of Two Lists"],
            problem_statement=problem_statement
        )