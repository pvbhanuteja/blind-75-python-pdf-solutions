"""
Unit tests for Blind 75 LeetCode solutions.
"""

import unittest
from typing import List, Optional
from blind_75_solutions.array_solutions import ArraySolutions
from blind_75_solutions.string_solutions import StringSolutions
from blind_75_solutions.linked_list_solutions import LinkedListSolutions, ListNode
from blind_75_solutions.tree_solutions import TreeSolutions, TreeNode
from blind_75_solutions.dynamic_programming_solutions import DynamicProgrammingSolutions
from blind_75_solutions.graph_solutions import GraphSolutions
from blind_75_solutions.interval_solutions import IntervalSolutions
from blind_75_solutions.matrix_solutions import MatrixSolutions
from blind_75_solutions.heap_solutions import HeapSolutions


class TestArraySolutions(unittest.TestCase):
    def setUp(self):
        self.array_solutions = ArraySolutions()

    def test_two_sum(self):
        solution = self.array_solutions.two_sum()
        self.assertEqual(solution.easy_solution([2, 7, 11, 15], 9), [0, 1])
        self.assertEqual(solution.optimized_solution([2, 7, 11, 15], 9), [0, 1])

    def test_best_time_to_buy_sell_stock(self):
        solution = self.array_solutions.best_time_to_buy_sell_stock()
        self.assertEqual(solution.easy_solution([7, 1, 5, 3, 6, 4]), 5)
        self.assertEqual(solution.optimized_solution([7, 1, 5, 3, 6, 4]), 5)

    def test_contains_duplicate(self):
        solution = self.array_solutions.contains_duplicate()
        self.assertTrue(solution.easy_solution([1, 2, 3, 1]))
        self.assertTrue(solution.optimized_solution([1, 2, 3, 1]))

    def test_product_of_array_except_self(self):
        solution = self.array_solutions.product_of_array_except_self()
        self.assertEqual(solution.easy_solution([1, 2, 3, 4]), [24, 12, 8, 6])
        self.assertEqual(solution.optimized_solution([1, 2, 3, 4]), [24, 12, 8, 6])

    def test_maximum_subarray(self):
        solution = self.array_solutions.maximum_subarray()
        self.assertEqual(solution.easy_solution([-2, 1, -3, 4, -1, 2, 1, -5, 4]), 6)
        self.assertEqual(
            solution.optimized_solution([-2, 1, -3, 4, -1, 2, 1, -5, 4]), 6
        )

    def test_maximum_product_subarray(self):
        solution = self.array_solutions.maximum_product_subarray()
        self.assertEqual(solution.easy_solution([2, 3, -2, 4]), 6)
        self.assertEqual(solution.optimized_solution([2, 3, -2, 4]), 6)

    def test_find_minimum_in_rotated_sorted_array(self):
        solution = self.array_solutions.find_minimum_in_rotated_sorted_array()
        self.assertEqual(solution.easy_solution([3, 4, 5, 1, 2]), 1)
        self.assertEqual(solution.optimized_solution([3, 4, 5, 1, 2]), 1)

    def test_search_in_rotated_sorted_array(self):
        solution = self.array_solutions.search_in_rotated_sorted_array()
        self.assertEqual(solution.easy_solution([4, 5, 6, 7, 0, 1, 2], 0), 4)
        self.assertEqual(solution.optimized_solution([4, 5, 6, 7, 0, 1, 2], 0), 4)

    def test_3sum(self):
        solution = self.array_solutions.three_sum()
        self.assertEqual(
            solution.easy_solution([-1, 0, 1, 2, -1, -4]), [[-1, -1, 2], [-1, 0, 1]]
        )
        self.assertEqual(
            solution.optimized_solution([-1, 0, 1, 2, -1, -4]),
            [[-1, -1, 2], [-1, 0, 1]],
        )

    def test_container_with_most_water(self):
        solution = self.array_solutions.container_with_most_water()
        self.assertEqual(solution.easy_solution([1, 8, 6, 2, 5, 4, 8, 3, 7]), 49)
        self.assertEqual(solution.optimized_solution([1, 8, 6, 2, 5, 4, 8, 3, 7]), 49)


class TestStringSolutions(unittest.TestCase):
    def setUp(self):
        self.string_solutions = StringSolutions()

    def test_longest_substring_without_repeating_characters(self):
        solution = (
            self.string_solutions.longest_substring_without_repeating_characters()
        )
        self.assertEqual(solution.easy_solution("abcabcbb"), 3)
        self.assertEqual(solution.optimized_solution("abcabcbb"), 3)

    def test_longest_repeating_character_replacement(self):
        solution = self.string_solutions.longest_repeating_character_replacement()
        self.assertEqual(solution.easy_solution("AABABBA", 1), 4)
        self.assertEqual(solution.optimized_solution("AABABBA", 1), 4)

    def test_minimum_window_substring(self):
        solution = self.string_solutions.minimum_window_substring()
        self.assertEqual(solution.easy_solution("ADOBECODEBANC", "ABC"), "BANC")
        self.assertEqual(solution.optimized_solution("ADOBECODEBANC", "ABC"), "BANC")

    def test_valid_anagram(self):
        solution = self.string_solutions.valid_anagram()
        self.assertTrue(solution.easy_solution("anagram", "nagaram"))
        self.assertTrue(solution.optimized_solution("anagram", "nagaram"))

    def test_group_anagrams(self):
        solution = self.string_solutions.group_anagrams()
        self.assertEqual(
            solution.easy_solution(["eat", "tea", "tan", "ate", "nat", "bat"]),
            [["eat", "tea", "ate"], ["tan", "nat"], ["bat"]],
        )
        self.assertEqual(
            solution.optimized_solution(["eat", "tea", "tan", "ate", "nat", "bat"]),
            [["eat", "tea", "ate"], ["tan", "nat"], ["bat"]],
        )

    def test_valid_parentheses(self):
        solution = self.string_solutions.valid_parentheses()
        self.assertTrue(solution.easy_solution("()[]{}"))
        self.assertTrue(solution.optimized_solution("()[]{}"))

    def test_valid_palindrome(self):
        solution = self.string_solutions.valid_palindrome()
        self.assertTrue(solution.easy_solution("A man, a plan, a canal: Panama"))
        self.assertTrue(solution.optimized_solution("A man, a plan, a canal: Panama"))

    def test_longest_palindromic_substring(self):
        solution = self.string_solutions.longest_palindromic_substring()
        self.assertEqual(solution.easy_solution("babad"), "bab")
        self.assertEqual(solution.optimized_solution("babad"), "bab")

    def test_palindromic_substrings(self):
        solution = self.string_solutions.palindromic_substrings()
        self.assertEqual(solution.easy_solution("abc"), 3)
        self.assertEqual(solution.optimized_solution("abc"), 3)

    def test_encode_and_decode_strings(self):
        solution = self.string_solutions.encode_and_decode_strings()
        original = ["Hello", "World"]
        codec = solution.easy_solution()
        encoded = codec.encode(original)
        self.assertEqual(codec.decode(encoded), original)
        codec = solution.optimized_solution()
        encoded = codec.encode(original)
        self.assertEqual(codec.decode(encoded), original)


class TestLinkedListSolutions(unittest.TestCase):
    def setUp(self):
        self.linked_list_solutions = LinkedListSolutions()

    def create_linked_list(self, values: List[int]) -> Optional[ListNode]:
        dummy = ListNode(0)
        current = dummy
        for val in values:
            current.next = ListNode(val)
            current = current.next
        return dummy.next

    def linked_list_to_list(self, head: Optional[ListNode]) -> List[int]:
        result = []
        current = head
        while current:
            result.append(current.val)
            current = current.next
        return result

    def test_reverse_linked_list(self):
        solution = self.linked_list_solutions.reverse_linked_list()
        head = self.create_linked_list([1, 2, 3, 4, 5])
        reversed_head = solution.easy_solution(head)
        self.assertEqual(self.linked_list_to_list(reversed_head), [5, 4, 3, 2, 1])
        head = self.create_linked_list([1, 2, 3, 4, 5])
        reversed_head = solution.optimized_solution(head)
        self.assertEqual(self.linked_list_to_list(reversed_head), [5, 4, 3, 2, 1])

    def test_linked_list_cycle(self):
        solution = self.linked_list_solutions.linked_list_cycle()
        head = self.create_linked_list([3, 2, 0, -4])
        head.next.next.next.next = head.next
        self.assertTrue(solution.easy_solution(head))
        self.assertTrue(solution.optimized_solution(head))

    def test_merge_two_sorted_lists(self):
        solution = self.linked_list_solutions.merge_two_sorted_lists()
        l1 = self.create_linked_list([1, 2, 4])
        l2 = self.create_linked_list([1, 3, 4])
        merged = solution.easy_solution(l1, l2)
        self.assertEqual(self.linked_list_to_list(merged), [1, 1, 2, 3, 4, 4])
        l1 = self.create_linked_list([1, 2, 4])
        l2 = self.create_linked_list([1, 3, 4])
        merged = solution.optimized_solution(l1, l2)
        self.assertEqual(self.linked_list_to_list(merged), [1, 1, 2, 3, 4, 4])

    def test_remove_nth_node_from_end_of_list(self):
        solution = self.linked_list_solutions.remove_nth_node_from_end_of_list()
        head = self.create_linked_list([1, 2, 3, 4, 5])
        result = solution.easy_solution(head, 2)
        self.assertEqual(self.linked_list_to_list(result), [1, 2, 3, 5])
        head = self.create_linked_list([1, 2, 3, 4, 5])
        result = solution.optimized_solution(head, 2)
        self.assertEqual(self.linked_list_to_list(result), [1, 2, 3, 5])

    def test_reorder_list(self):
        solution = self.linked_list_solutions.reorder_list()
        head = self.create_linked_list([1, 2, 3, 4])
        solution.easy_solution(head)
        self.assertEqual(self.linked_list_to_list(head), [1, 4, 2, 3])
        head = self.create_linked_list([1, 2, 3, 4])
        solution.optimized_solution(head)
        self.assertEqual(self.linked_list_to_list(head), [1, 4, 2, 3])


class TestTreeSolutions(unittest.TestCase):
    def setUp(self):
        self.tree_solutions = TreeSolutions()

    def create_tree(self, values: List[Optional[int]]) -> Optional[TreeNode]:
        if not values:
            return None
        nodes = [TreeNode(val) if val is not None else None for val in values]
        for i in range(len(nodes)):
            if nodes[i]:
                left = 2 * i + 1
                right = 2 * i + 2
                if left < len(nodes):
                    nodes[i].left = nodes[left]
                if right < len(nodes):
                    nodes[i].right = nodes[right]
        return nodes[0]

    def test_maximum_depth_of_binary_tree(self):
        solution = self.tree_solutions.maximum_depth_of_binary_tree()
        root = self.create_tree([3, 9, 20, None, None, 15, 7])
        self.assertEqual(solution.easy_solution(root), 3)
        self.assertEqual(solution.optimized_solution(root), 3)

    def test_same_tree(self):
        solution = self.tree_solutions.same_tree()
        p = self.create_tree([1, 2, 3])
        q = self.create_tree([1, 2, 3])
        self.assertTrue(solution.easy_solution(p, q))
        self.assertTrue(solution.optimized_solution(p, q))

    def test_invert_binary_tree(self):
        solution = self.tree_solutions.invert_binary_tree()
        root = self.create_tree([4, 2, 7, 1, 3, 6, 9])
        inverted = solution.easy_solution(root)
        self.assertEqual(inverted.left.val, 7)
        self.assertEqual(inverted.right.val, 2)
        root = self.create_tree([4, 2, 7, 1, 3, 6, 9])
        inverted = solution.optimized_solution(root)
        self.assertEqual(inverted.left.val, 7)
        self.assertEqual(inverted.right.val, 2)

    def test_binary_tree_maximum_path_sum(self):
        solution = self.tree_solutions.binary_tree_maximum_path_sum()
        root = self.create_tree([1, 2, 3])
        self.assertEqual(solution.easy_solution(root), 6)
        self.assertEqual(solution.optimized_solution(root), 6)

    def test_binary_tree_level_order_traversal(self):
        solution = self.tree_solutions.binary_tree_level_order_traversal()
        root = self.create_tree([3, 9, 20, None, None, 15, 7])
        self.assertEqual(solution.easy_solution(root), [[3], [9, 20], [15, 7]])
        self.assertEqual(solution.optimized_solution(root), [[3], [9, 20], [15, 7]])

    def test_serialize_and_deserialize_binary_tree(self):
        solution = self.tree_solutions.serialize_and_deserialize_binary_tree()
        root = self.create_tree([1, 2, 3, None, None, 4, 5])
        codec = solution.easy_solution()
        serialized = codec.serialize(root)
        deserialized = codec.deserialize(serialized)
        self.assertEqual(solution.easy_solution(root, deserialized), True)
        codec = solution.optimized_solution()
        serialized = codec.serialize(root)
        deserialized = codec.deserialize(serialized)
        self.assertEqual
