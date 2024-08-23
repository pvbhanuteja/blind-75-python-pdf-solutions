## blind_75_solutions/tree_solutions.py

"""
Module containing tree-related solutions for Blind 75 LeetCode questions.
"""

from typing import Optional, List
from blind_75_solutions import Solution


class TreeNode:
    """Definition for a binary tree node."""

    def __init__(
        self,
        val: int = 0,
        left: Optional["TreeNode"] = None,
        right: Optional["TreeNode"] = None,
    ):
        self.val = val
        self.left = left
        self.right = right


class TreeSolutions:
    """Class containing tree-related solutions for Blind 75 LeetCode questions."""

    @staticmethod
    def maximum_depth_of_binary_tree() -> Solution:
        """
        Generate a Solution object for the Maximum Depth of Binary Tree problem.

        Returns:
            Solution: A Solution object containing details of the problem.
        """

        def problem_statement() -> str:
            return """Given the root of a binary tree, return its maximum depth.

A binary tree's maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.

Example 1:
Input: root = [3,9,20,null,null,15,7]
Output: 3

Example 2:
Input: root = [1,null,2]
Output: 2

Constraints:
    The number of nodes in the tree is in the range [0, 104].
    -100 <= Node.val <= 100

https://leetcode.com/problems/maximum-depth-of-binary-tree/description/
"""

        class TreeNode:
            def __init__(self, val=0, left=None, right=None):
                self.val = val
                self.left = left
                self.right = right

        def easy_solution(root: TreeNode) -> int:
            # Recursive solution: Calculate depth for each subtree
            if not root:
                return 0
            return 1 + max(easy_solution(root.left), easy_solution(root.right))

        def optimized_solution(root: TreeNode) -> int:
            # Iterative solution: Use depth-first search with stack
            if not root:
                return 0

            stack = [(root, 1)]
            max_depth = 0

            while stack:
                node, depth = stack.pop()
                max_depth = max(max_depth, depth)

                if node.right:
                    stack.append((node.right, depth + 1))
                if node.left:
                    stack.append((node.left, depth + 1))

            return max_depth

        return Solution(
            question="Maximum Depth of Binary Tree",
            easy_solution=easy_solution,
            optimized_solution=optimized_solution,
            time_complexity="O(n)",
            space_complexity="O(h)",
            similar_questions=["Balanced Binary Tree", "Minimum Depth of Binary Tree"],
            problem_statement=problem_statement,
        )

    @staticmethod
    def same_tree() -> Solution:
        """
        Generate a Solution object for the Same Tree problem.

        Returns:
            Solution: A Solution object containing details of the problem.
        """

        def problem_statement() -> str:
            return """Given the roots of two binary trees p and q, write a function to check if they are the same or not.

Two binary trees are considered the same if they are structurally identical, and the nodes have the same value.

Example 1:
Input: p = [1,2,3], q = [1,2,3]
Output: true

Example 2:
Input: p = [1,2], q = [1,null,2]
Output: false

Example 3:
Input: p = [1,2,1], q = [1,1,2]
Output: false

Constraints:
    The number of nodes in both trees is in the range [0, 100].
    -104 <= Node.val <= 104

https://leetcode.com/problems/same-tree/description/
"""

        class TreeNode:
            def __init__(self, val=0, left=None, right=None):
                self.val = val
                self.left = left
                self.right = right

        def easy_solution(p: TreeNode, q: TreeNode) -> bool:
            # Recursive solution: Check each node
            if not p and not q:
                return True
            if not p or not q:
                return False
            return (
                p.val == q.val
                and easy_solution(p.left, q.left)
                and easy_solution(p.right, q.right)
            )

        def optimized_solution(p: TreeNode, q: TreeNode) -> bool:
            # Iterative solution: Use stack to compare nodes
            stack = [(p, q)]
            while stack:
                node1, node2 = stack.pop()
                if not node1 and not node2:
                    continue
                if not node1 or not node2:
                    return False
                if node1.val != node2.val:
                    return False
                stack.append((node1.right, node2.right))
                stack.append((node1.left, node2.left))
            return True

        return Solution(
            question="Same Tree",
            easy_solution=easy_solution,
            optimized_solution=optimized_solution,
            time_complexity="O(min(n, m))",
            space_complexity="O(min(h1, h2))",
            similar_questions=["Symmetric Tree", "Subtree of Another Tree"],
            problem_statement=problem_statement,
        )

    @staticmethod
    def invert_binary_tree() -> Solution:
        """
        Generate a Solution object for the Invert Binary Tree problem.

        Returns:
            Solution: A Solution object containing details of the problem.
        """

        def problem_statement() -> str:
            return """Given the root of a binary tree, invert the tree, and return its root.

Example 1:
Input: root = [4,2,7,1,3,6,9]
Output: [4,7,2,9,6,3,1]

Example 2:
Input: root = [2,1,3]
Output: [2,3,1]

Example 3:
Input: root = []
Output: []

Constraints:
    The number of nodes in the tree is in the range [0, 100].
    -100 <= Node.val <= 100

https://leetcode.com/problems/invert-binary-tree/description/
"""

        class TreeNode:
            def __init__(self, val=0, left=None, right=None):
                self.val = val
                self.left = left
                self.right = right

        def easy_solution(root: TreeNode) -> TreeNode:
            # Recursive solution: Swap left and right subtrees
            if not root:
                return None
            root.left, root.right = easy_solution(root.right), easy_solution(root.left)
            return root

        def optimized_solution(root: TreeNode) -> TreeNode:
            # Iterative solution: Use queue to invert the tree
            if not root:
                return None

            queue = [root]
            while queue:
                node = queue.pop(0)
                node.left, node.right = node.right, node.left

                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)

            return root

        return Solution(
            question="Invert Binary Tree",
            easy_solution=easy_solution,
            optimized_solution=optimized_solution,
            time_complexity="O(n)",
            space_complexity="O(h)",
            similar_questions=["Reverse Odd Levels of Binary Tree"],
            problem_statement=problem_statement,
        )

    @staticmethod
    def binary_tree_maximum_path_sum() -> Solution:
        """
        Generate a Solution object for the Binary Tree Maximum Path Sum problem.

        Returns:
            Solution: A Solution object containing details of the Binary Tree Maximum Path Sum problem.
        """

        def problem_statement() -> str:
            return """A path in a binary tree is a sequence of nodes where each pair of adjacent nodes in the sequence has an edge connecting them. A node can only appear in the sequence at most once. Note that the path does not need to pass through the root.

The path sum of a path is the sum of the node's values in the path.

Given the root of a binary tree, return the maximum path sum of any path.

Example 1:

Input: root = [1,2,3]
Output: 6

Example 2:

Input: root = [-10,9,20,null,null,15,7]
Output: 42

Constraints:

    The number of nodes in the tree is in the range [1, 3 * 104].
    -1000 <= Node.val <= 1000

https://leetcode.com/problems/binary-tree-maximum-path-sum/
"""

        def easy_solution(root: Optional[TreeNode]) -> int:
            # Helper function to calculate the maximum gain from each node
            def max_gain(node: Optional[TreeNode]) -> int:
                nonlocal max_sum
                if not node:
                    return 0
                left_gain = max(max_gain(node.left), 0)
                right_gain = max(max_gain(node.right), 0)
                path_sum = node.val + left_gain + right_gain
                max_sum = max(max_sum, path_sum)
                return node.val + max(left_gain, right_gain)

            max_sum = float("-inf")
            max_gain(root)
            return max_sum

        def optimized_solution(root: Optional[TreeNode]) -> int:
            # The easy solution is already optimal for this problem
            return easy_solution(root)

        return Solution(
            question="Binary Tree Maximum Path Sum",
            easy_solution=easy_solution,
            optimized_solution=optimized_solution,
            time_complexity="O(n)",
            space_complexity="O(h)",
            similar_questions=["Path Sum", "Sum Root to Leaf Numbers"],
            problem_statement=problem_statement,
        )

    @staticmethod
    def binary_tree_level_order_traversal() -> Solution:
        """
        Generate a Solution object for the Binary Tree Level Order Traversal problem.

        Returns:
            Solution: A Solution object containing details of the Binary Tree Level Order Traversal problem.
        """

        def problem_statement() -> str:
            return """Given the root of a binary tree, return the level order traversal of its nodes' values. (i.e., from left to right, level by level).

Example 1:

Input: root = [3,9,20,null,null,15,7]
Output: [[3],[9,20],[15,7]]

Example 2:

Input: root = [1]
Output: [[1]]

Example 3:

Input: root = []
Output: []

Constraints:

    The number of nodes in the tree is in the range [0, 2000].
    -1000 <= Node.val <= 1000

https://leetcode.com/problems/binary-tree-level-order-traversal/
"""

        def easy_solution(root: Optional[TreeNode]) -> List[List[int]]:
            # Iterative solution: Use a queue to traverse each level
            if not root:
                return []
            result = []
            queue = [root]
            while queue:
                level_size = len(queue)
                level = []
                for _ in range(level_size):
                    node = queue.pop(0)
                    level.append(node.val)
                    if node.left:
                        queue.append(node.left)
                    if node.right:
                        queue.append(node.right)
                result.append(level)
            return result

        def optimized_solution(root: Optional[TreeNode]) -> List[List[int]]:
            # More optimized iterative solution: Use a queue to traverse each level
            if not root:
                return []
            result = []
            level = [root]
            while level:
                result.append([node.val for node in level])
                level = [
                    child
                    for node in level
                    for child in (node.left, node.right)
                    if child
                ]
            return result

        return Solution(
            question="Binary Tree Level Order Traversal",
            easy_solution=easy_solution,
            optimized_solution=optimized_solution,
            time_complexity="O(n)",
            space_complexity="O(n)",
            similar_questions=[
                "Binary Tree Zigzag Level Order Traversal",
                "Binary Tree Level Order Traversal II",
                "Minimum Depth of Binary Tree",
                "Binary Tree Vertical Order Traversal",
            ],
            problem_statement=problem_statement,
        )

    @staticmethod
    def serialize_and_deserialize_binary_tree() -> Solution:
        """
        Generate a Solution object for the Serialize and Deserialize Binary Tree problem.

        Returns:
            Solution: A Solution object containing details of the Serialize and Deserialize Binary Tree problem.
        """

        def problem_statement() -> str:
            return """Serialization is the process of converting a data structure or object into a sequence of bits so that it can be stored in a file or memory buffer, or transmitted across a network connection link to be reconstructed later in the same or another computer environment.

Design an algorithm to serialize and deserialize a binary tree. There is no restriction on how your serialization/deserialization algorithm should work. You just need to ensure that a binary tree can be serialized to a string and this string can be deserialized to the original tree structure.

Clarification: The input/output format is the same as how LeetCode serializes a binary tree. You do not necessarily need to follow this format, so please be creative and come up with different approaches yourself.

Example 1:

Input: root = [1,2,3,null,null,4,5]
Output: [1,2,3,null,null,4,5]

Example 2:

Input: root = []
Output: []

Constraints:

    The number of nodes in the tree is in the range [0, 104].
    -1000 <= Node.val <= 1000

https://leetcode.com/problems/serialize-and-deserialize-binary-tree/
"""

        class Codec:
            def serialize(self, root: Optional[TreeNode]) -> str:
                """Serializes a tree to a single string."""
                if not root:
                    return "null"
                return f"{root.val},{self.serialize(root.left)},{self.serialize(root.right)}"

            def deserialize(self, data: str) -> Optional[TreeNode]:
                """Deserializes your encoded data to tree."""

                def dfs() -> Optional[TreeNode]:
                    val = next(values)
                    if val == "null":
                        return None
                    node = TreeNode(int(val))
                    node.left = dfs()
                    node.right = dfs()
                    return node

                values = iter(data.split(","))
                return dfs()

        def easy_solution(root: Optional[TreeNode]) -> Optional[TreeNode]:
            # Use Codec class to serialize and deserialize
            codec = Codec()
            serialized = codec.serialize(root)
            return codec.deserialize(serialized)

        def optimized_solution(root: Optional[TreeNode]) -> Optional[TreeNode]:
            # The easy solution is already optimal for this problem
            return easy_solution(root)

        return Solution(
            question="Serialize and Deserialize Binary Tree",
            easy_solution=easy_solution,
            optimized_solution=optimized_solution,
            time_complexity="O(n)",
            space_complexity="O(n)",
            similar_questions=[
                "Encode and Decode Strings",
                "Serialize and Deserialize BST",
                "Find Duplicate Subtrees",
                "Serialize and Deserialize N-ary Tree",
            ],
            problem_statement=problem_statement,
        )

    @staticmethod
    def subtree_of_another_tree() -> Solution:
        """
        Generate a Solution object for the Subtree of Another Tree problem.

        Returns:
            Solution: A Solution object containing details of the Subtree of Another Tree problem.
        """

        def problem_statement() -> str:
            return """Given the roots of two binary trees root and subRoot, return true if there is a subtree of root with the same structure and node values of subRoot and false otherwise.

A subtree of a binary tree tree is a tree that consists of a node in tree and all of this node's descendants. The tree tree could also be considered as a subtree of itself.

Example 1:

Input: root = [3,4,5,1,2], subRoot = [4,1,2]
Output: true

Example 2:

Input: root = [3,4,5,1,2,null,null,null,null,0], subRoot = [4,1,2]
Output: false

Constraints:

    The number of nodes in the root tree is in the range [1, 2000].
    The number of nodes in the subRoot tree is in the range [1, 1000].
    -104 <= root.val <= 104
    -104 <= subRoot.val <= 104

https://leetcode.com/problems/subtree-of-another-tree/
"""

        def is_same_tree(s: Optional[TreeNode], t: Optional[TreeNode]) -> bool:
            # Helper function to check if two trees are the same
            if not s and not t:
                return True
            if not s or not t:
                return False
            return (
                s.val == t.val
                and is_same_tree(s.left, t.left)
                and is_same_tree(s.right, t.right)
            )

        def easy_solution(
            root: Optional[TreeNode], subRoot: Optional[TreeNode]
        ) -> bool:
            # Recursive solution: Check each subtree
            if not root:
                return False
            if is_same_tree(root, subRoot):
                return True
            return easy_solution(root.left, subRoot) or easy_solution(
                root.right, subRoot
            )

        def optimized_solution(
            root: Optional[TreeNode], subRoot: Optional[TreeNode]
        ) -> bool:
            # Optimized solution: Serialize trees and use KMP algorithm to find substring
            def serialize(node: Optional[TreeNode]) -> str:
                if not node:
                    return "#"
                return f"{node.val},{serialize(node.left)},{serialize(node.right)}"

            def kmp_search(text: str, pattern: str) -> bool:
                if not pattern:
                    return True
                lps = [0] * len(pattern)
                length = 0
                i = 1
                while i < len(pattern):
                    if pattern[i] == pattern[length]:
                        length += 1
                        lps[i] = length
                        i += 1
                    elif length != 0:
                        length = lps[length - 1]
                    else:
                        lps[i] = 0
                        i += 1

                i = j = 0
                while i < len(text):
                    if pattern[j] == text[i]:
                        i += 1
                        j += 1
                    if j == len(pattern):
                        return True
                    elif i < len(text) and pattern[j] != text[i]:
                        if j != 0:
                            j = lps[j - 1]
                        else:
                            i += 1
                return False

            return kmp_search(serialize(root), serialize(subRoot))

        return Solution(
            question="Subtree of Another Tree",
            easy_solution=easy_solution,
            optimized_solution=optimized_solution,
            time_complexity="O(m + n)",
            space_complexity="O(m + n)",
            similar_questions=["Count Univalue Subtrees", "Most Frequent Subtree Sum"],
            problem_statement=problem_statement,
        )

    @staticmethod
    def construct_binary_tree_from_preorder_and_inorder_traversal() -> Solution:
        """
        Generate a Solution object for the Construct Binary Tree from Preorder and Inorder Traversal problem.

        Returns:
            Solution: A Solution object containing details of the Construct Binary Tree from Preorder and Inorder Traversal problem.
        """

        def problem_statement() -> str:
            return """Given two integer arrays preorder and inorder where preorder is the preorder traversal of a binary tree and inorder is the inorder traversal of the same tree, construct and return the binary tree.

Example 1:

Input: preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
Output: [3,9,20,null,null,15,7]

Example 2:

Input: preorder = [-1], inorder = [-1]
Output: [-1]

Constraints:

    1 <= preorder.length <= 3000
    inorder.length == preorder.length
    -3000 <= preorder[i], inorder[i] <= 3000
    preorder and inorder consist of unique values.
    Each value of inorder also appears in preorder.
    preorder is guaranteed to be the preorder traversal of the tree.
    inorder is guaranteed to be the inorder traversal of the tree.

https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/
"""

        def easy_solution(
            preorder: List[int], inorder: List[int]
        ) -> Optional[TreeNode]:
            # Recursive solution: Use preorder and inorder to build the tree
            if not preorder or not inorder:
                return None
            root = TreeNode(preorder[0])
            mid = inorder.index(preorder[0])
            root.left = easy_solution(preorder[1 : mid + 1], inorder[:mid])
            root.right = easy_solution(preorder[mid + 1 :], inorder[mid + 1 :])
            return root

        def optimized_solution(
            preorder: List[int], inorder: List[int]
        ) -> Optional[TreeNode]:
            # Optimized solution: Use a hashmap to quickly find the root in inorder
            def build(start: int, end: int) -> Optional[TreeNode]:
                nonlocal pre_idx
                if start > end:
                    return None
                root = TreeNode(preorder[pre_idx])
                pre_idx += 1
                mid = inorder_map[root.val]
                root.left = build(start, mid - 1)
                root.right = build(mid + 1, end)
                return root

            pre_idx = 0
            inorder_map = {val: idx for idx, val in enumerate(inorder)}
            return build(0, len(inorder) - 1)

        return Solution(
            question="Construct Binary Tree from Preorder and Inorder Traversal",
            easy_solution=easy_solution,
            optimized_solution=optimized_solution,
            time_complexity="O(n)",
            space_complexity="O(n)",
            similar_questions=[
                "Construct Binary Tree from Inorder and Postorder Traversal"
            ],
            problem_statement=problem_statement,
        )

    @staticmethod
    def validate_binary_search_tree() -> Solution:
        """
        Generate a Solution object for the Validate Binary Search Tree problem.

        Returns:
            Solution: A Solution object containing details of the Validate Binary Search Tree problem.
        """

        def problem_statement() -> str:
            return """Given the root of a binary tree, determine if it is a valid binary search tree (BST).

A valid BST is defined as follows:

The left subtree of a node contains only nodes with keys less than the node's key.
The right subtree of a node contains only nodes with keys greater than the node's key.
Both the left and right subtrees must also be binary search trees.

Example 1:

Input: root = [2,1,3]
Output: true

Example 2:

Input: root = [5,1,4,null,null,3,6]
Output: false
Explanation: The root node's value is 5 but its right child's value is 4.

Constraints:

    The number of nodes in the tree is in the range [1, 104].
    -231 <= Node.val <= 231 - 1

https://leetcode.com/problems/validate-binary-search-tree/
"""

        def easy_solution(root: Optional[TreeNode]) -> bool:
            # Recursive solution: Check each node's value within valid range
            def is_valid_bst(
                node: Optional[TreeNode], min_val: float, max_val: float
            ) -> bool:
                if not node:
                    return True
                if node.val <= min_val or node.val >= max_val:
                    return False
                return is_valid_bst(node.left, min_val, node.val) and is_valid_bst(
                    node.right, node.val, max_val
                )

            return is_valid_bst(root, float("-inf"), float("inf"))

        def optimized_solution(root: Optional[TreeNode]) -> bool:
            # Iterative solution: Use a stack to check nodes
            stack = [(root, float("-inf"), float("inf"))]
            while stack:
                node, min_val, max_val = stack.pop()
                if not node:
                    continue
                if node.val <= min_val or node.val >= max_val:
                    return False
                stack.append((node.right, node.val, max_val))
                stack.append((node.left, min_val, node.val))
            return True

        return Solution(
            question="Validate Binary Search Tree",
            easy_solution=easy_solution,
            optimized_solution=optimized_solution,
            time_complexity="O(n)",
            space_complexity="O(h)",
            similar_questions=[
                "Binary Tree Inorder Traversal",
                "Find Mode in Binary Search Tree",
            ],
            problem_statement=problem_statement,
        )

    @staticmethod
    def kth_smallest_element_in_a_bst() -> Solution:
        """
        Generate a Solution object for the Kth Smallest Element in a BST problem.

        Returns:
            Solution: A Solution object containing details of the Kth Smallest Element in a BST problem.
        """

        def problem_statement() -> str:
            return """Given the root of a binary search tree, and an integer k, return the kth smallest value (1-indexed) of all the values of the nodes in the tree.

Example 1:

Input: root = [3,1,4,null,2], k = 1
Output: 1

Example 2:

Input: root = [5,3,6,2,4,null,null,1], k = 3
Output: 3

Constraints:

    The number of nodes in the tree is n.
    1 <= k <= n <= 104
    0 <= Node.val <= 104

https://leetcode.com/problems/kth-smallest-element-in-a-bst/
"""

        def easy_solution(root: Optional[TreeNode], k: int) -> int:
            # Recursive solution: Inorder traversal to find kth smallest
            def inorder(node: Optional[TreeNode]) -> None:
                nonlocal k, result
                if not node:
                    return
                inorder(node.left)
                k -= 1
                if k == 0:
                    result = node.val
                    return
                inorder(node.right)

            result = None
            inorder(root)
            return result

        def optimized_solution(root: Optional[TreeNode], k: int) -> int:
            # Iterative solution: Inorder traversal using stack
            stack = []
            current = root

            while current or stack:
                while current:
                    stack.append(current)
                    current = current.left

                current = stack.pop()
                k -= 1
                if k == 0:
                    return current.val

                current = current.right

        return Solution(
            question="Kth Smallest Element in a BST",
            easy_solution=easy_solution,
            optimized_solution=optimized_solution,
            time_complexity="O(H + k)",
            space_complexity="O(H)",
            similar_questions=[
                "Binary Tree Inorder Traversal",
                "Second Minimum Node In a Binary Tree",
            ],
            problem_statement=problem_statement,
        )

    @staticmethod
    def lowest_common_ancestor_of_a_binary_search_tree() -> Solution:
        """
        Generate a Solution object for the Lowest Common Ancestor of a Binary Search Tree problem.

        Returns:
            Solution: A Solution object containing details of the Lowest Common Ancestor of a Binary Search Tree problem.
        """

        def problem_statement() -> str:
            return """Given a binary search tree (BST), find the lowest common ancestor (LCA) node of two given nodes in the BST.

According to the definition of LCA on Wikipedia: “The lowest common ancestor is defined between two nodes p and q as the lowest node in T that has both p and q as descendants (where we allow a node to be a descendant of itself).”

Example 1:

Input: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 8
Output: 6
Explanation: The LCA of nodes 2 and 8 is 6.

Example 2:

Input: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 4
Output: 2
Explanation: The LCA of nodes 2 and 4 is 2, since a node can be a descendant of itself according to the LCA definition.

Example 3:

Input: root = [2,1], p = 2, q = 1
Output: 2

Constraints:

    The number of nodes in the tree is in the range [2, 105].
    -109 <= Node.val <= 109
    All Node.val are unique.
    p != q
    p and q will exist in the BST.

https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/
"""

        def easy_solution(
            root: Optional[TreeNode], p: TreeNode, q: TreeNode
        ) -> Optional[TreeNode]:
            # Recursive solution: Traverse the tree to find LCA
            if not root:
                return None
            if p.val < root.val and q.val < root.val:
                return easy_solution(root.left, p, q)
            if p.val > root.val and q.val > root.val:
                return easy_solution(root.right, p, q)
            return root

        def optimized_solution(
            root: Optional[TreeNode], p: TreeNode, q: TreeNode
        ) -> Optional[TreeNode]:
            # Iterative solution: Traverse the tree to find LCA
            current = root
            while current:
                if p.val < current.val and q.val < current.val:
                    current = current.left
                elif p.val > current.val and q.val > current.val:
                    current = current.right
                else:
                    return current

        return Solution(
            question="Lowest Common Ancestor of a Binary Search Tree",
            easy_solution=easy_solution,
            optimized_solution=optimized_solution,
            time_complexity="O(H)",
            space_complexity="O(1)",
            similar_questions=[
                "Lowest Common Ancestor of a Binary Tree",
                "Smallest Common Region",
            ],
            problem_statement=problem_statement,
        )

    @staticmethod
    def implement_trie_prefix_tree() -> Solution:
        """
        Generate a Solution object for the Implement Trie (Prefix Tree) problem.

        Returns:
            Solution: A Solution object containing details of the Implement Trie (Prefix Tree) problem.
        """

        def problem_statement() -> str:
            return """A trie (pronounced as "try") or prefix tree is a tree data structure used to efficiently store and retrieve keys in a dataset of strings. There are various applications of this data structure, such as autocomplete and spellchecker.

Implement the Trie class:

    Trie() Initializes the trie object.
    void insert(String word) Inserts the string word into the trie.
    boolean search(String word) Returns true if the string word is in the trie (i.e., was inserted before), and false otherwise.
    boolean startsWith(String prefix) Returns true if there is a previously inserted string word that has the prefix prefix, and false otherwise.

Example 1:

Input
["Trie", "insert", "search", "search", "startsWith", "insert", "search"]
[[], ["apple"], ["apple"], ["app"], ["app"], ["app"], ["app"]]
Output
[null, null, true, false, true, null, true]

Explanation
Trie trie = new Trie();
trie.insert("apple");
trie.search("apple");   // return True
trie.search("app");     // return False
trie.startsWith("app"); // return True
trie.insert("app");
trie.search("app");     // return True

Constraints:

    1 <= word.length, prefix.length <= 2000
    word and prefix consist only of lowercase English letters.
    At most 3 * 104 calls in total will be made to insert, search, and startsWith.

https://leetcode.com/problems/implement-trie-prefix-tree/
"""

        class TrieNode:
            def __init__(self):
                self.children = {}
                self.is_end = False

        class Trie:
            def __init__(self):
                self.root = TrieNode()

            def insert(self, word: str) -> None:
                node = self.root
                for char in word:
                    if char not in node.children:
                        node.children[char] = TrieNode()
                    node = node.children[char]
                node.is_end = True

            def search(self, word: str) -> bool:
                node = self.root
                for char in word:
                    if char not in node.children:
                        return False
                    node = node.children[char]
                return node.is_end

            def startsWith(self, prefix: str) -> bool:
                node = self.root
                for char in prefix:
                    if char not in node.children:
                        return False
                    node = node.children[char]
                return True

        def easy_solution():
            # Use the Trie class to perform operations
            return Trie()

        def optimized_solution():
            # The easy solution is already optimal for this problem
            return Trie()

        return Solution(
            question="Implement Trie (Prefix Tree)",
            easy_solution=easy_solution,
            optimized_solution=optimized_solution,
            time_complexity="O(m)",
            space_complexity="O(m)",
            similar_questions=[
                "Design Add and Search Words Data Structure",
                "Design Search Autocomplete System",
                "Replace Words",
                "Implement Magic Dictionary",
            ],
            problem_statement=problem_statement,
        )
