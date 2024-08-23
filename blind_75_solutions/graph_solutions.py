"""
Module containing graph-related solutions for Blind 75 LeetCode questions.
"""

from typing import List, Optional, Dict
from collections import deque
from blind_75_solutions import Solution


class GraphSolutions:
    """Class containing graph-related solutions for Blind 75 LeetCode questions."""

    @staticmethod
    def clone_graph() -> Solution:
        """
        Generate a Solution object for the Clone Graph problem.

        Returns:
            Solution: A Solution object containing details of the Clone Graph problem.
        """

        class Node:
            def __init__(self, val: int = 0, neighbors: Optional[List["Node"]] = None):
                self.val = val
                self.neighbors = neighbors if neighbors is not None else []

        def problem_statement() -> str:
            return """Given a reference of a node in a connected undirected graph.

Return a deep copy (clone) of the graph.

Each node in the graph contains a value (int) and a list (List[Node]) of its neighbors.

class Node {
    public int val;
    public List<Node> neighbors;
}

 

Test case format:

For simplicity, each node's value is the same as the node's index (1-indexed). For example, the first node with val == 1, the second node with val == 2, and so on. The graph is represented in the test case using an adjacency list.

An adjacency list is a collection of unordered lists used to represent a finite graph. Each list describes the set of neighbors of a node in the graph.

The given node will always be the first node with val = 1. You must return the copy of the given node as a reference to the cloned graph.

 

Example 1:

Input: adjList = [[2,4],[1,3],[2,4],[1,3]]
Output: [[2,4],[1,3],[2,4],[1,3]]
Explanation: There are 4 nodes in the graph.
1's neighbors are 2 and 4.
2's neighbors are 1 and 3.
3's neighbors are 2 and 4.
4's neighbors are 1 and 3.

Example 2:

Input: adjList = [[]]
Output: [[]]
Explanation: Note that the input contains one empty list. The graph consists of only one node with val = 1 since there are no neighbors.

Example 3:

Input: adjList = []
Output: []
Explanation: This an empty graph, it does not have any nodes.

 

Constraints:

    The number of nodes in the graph is in the range [0, 100].
    1 <= Node.val <= 100
    Node.val is unique for each node.
    There are no repeated edges and no self-loops in the graph.
    The Graph is connected and all nodes can be visited starting from the given node.

https://leetcode.com/problems/clone-graph/description/
"""

        def easy_solution(node: Optional[Node]) -> Optional[Node]:
            # Easy solution: DFS with recursion to clone the graph
            if not node:
                return None

            visited = {}

            def dfs(node: Node) -> Node:
                if node in visited:
                    return visited[node]

                clone = Node(node.val)
                visited[node] = clone

                for neighbor in node.neighbors:
                    clone.neighbors.append(dfs(neighbor))

                return clone

            return dfs(node)

        def optimized_solution(node: Optional[Node]) -> Optional[Node]:
            # Optimized solution: BFS with queue to clone the graph
            if not node:
                return None

            visited = {}
            queue = deque([node])
            visited[node] = Node(node.val)

            while queue:
                current = queue.popleft()
                for neighbor in current.neighbors:
                    if neighbor not in visited:
                        visited[neighbor] = Node(neighbor.val)
                        queue.append(neighbor)
                    visited[current].neighbors.append(visited[neighbor])

            return visited[node]

        return Solution(
            question="Clone Graph",
            easy_solution=easy_solution,
            optimized_solution=optimized_solution,
            time_complexity="O(N + E)",
            space_complexity="O(N)",
            similar_questions=[
                "Copy List with Random Pointer",
                "Clone Binary Tree With Random Pointer",
                "Clone N-ary Tree",
            ],
            problem_statement=problem_statement,
        )

    @staticmethod
    def course_schedule() -> Solution:
        """
        Generate a Solution object for the Course Schedule problem.

        Returns:
            Solution: A Solution object containing details of the Course Schedule problem.
        """

        def problem_statement() -> str:
            return """There are a total of numCourses courses you have to take, labeled from 0 to numCourses - 1. You are given an array prerequisites where prerequisites[i] = [ai, bi] indicates that you must take course bi first if you want to take course ai.

    For example, the pair [0, 1], indicates that to take course 0 you have to first take course 1.

Return true if you can finish all courses. Otherwise, return false.

 

Example 1:

Input: numCourses = 2, prerequisites = [[1,0]]
Output: true
Explanation: There are a total of 2 courses to take. To take course 1 you should have finished course 0. So it is possible.

Example 2:

Input: numCourses = 2, prerequisites = [[1,0],[0,1]]
Output: false
Explanation: There are a total of 2 courses to take. To take course 1 you should have finished course 0, and to take course 0 you should have finished course 1. So it is impossible.

 

Constraints:

    1 <= numCourses <= 2000
    0 <= prerequisites.length <= 5000
    prerequisites[i].length == 2
    0 <= ai, bi < numCourses
    All the pairs prerequisites[i] are unique.

https://leetcode.com/problems/course-schedule/description/
"""

        def easy_solution(numCourses: int, prerequisites: List[List[int]]) -> bool:
            # Easy solution: DFS to detect cycles in the graph
            graph = [[] for _ in range(numCourses)]
            for course, prereq in prerequisites:
                graph[course].append(prereq)

            def has_cycle(course: int, path: set) -> bool:
                if course in path:
                    return True
                if not graph[course]:
                    return False
                path.add(course)
                for prereq in graph[course]:
                    if has_cycle(prereq, path):
                        return True
                path.remove(course)
                graph[course] = []
                return False

            for course in range(numCourses):
                if has_cycle(course, set()):
                    return False
            return True

        def optimized_solution(numCourses: int, prerequisites: List[List[int]]) -> bool:
            # Optimized solution: Topological sort using Kahn's algorithm
            graph = [[] for _ in range(numCourses)]
            in_degree = [0] * numCourses

            for course, prereq in prerequisites:
                graph[prereq].append(course)
                in_degree[course] += 1

            queue = deque(
                [course for course in range(numCourses) if in_degree[course] == 0]
            )
            taken = 0

            while queue:
                course = queue.popleft()
                taken += 1
                for next_course in graph[course]:
                    in_degree[next_course] -= 1
                    if in_degree[next_course] == 0:
                        queue.append(next_course)

            return taken == numCourses

        return Solution(
            question="Course Schedule",
            easy_solution=easy_solution,
            optimized_solution=optimized_solution,
            time_complexity="O(V + E)",
            space_complexity="O(V + E)",
            similar_questions=[
                "Course Schedule II",
                "Graph Valid Tree",
                "Minimum Height Trees",
                "Course Schedule III",
            ],
            problem_statement=problem_statement,
        )

    @staticmethod
    def pacific_atlantic_water_flow() -> Solution:
        """
        Generate a Solution object for the Pacific Atlantic Water Flow problem.

        Returns:
            Solution: A Solution object containing details of the Pacific Atlantic Water Flow problem.
        """

        def problem_statement() -> str:
            return """There is an m x n rectangular island that borders both the Pacific Ocean and Atlantic Ocean. The Pacific Ocean touches the island's left and top edges, and the Atlantic Ocean touches the island's right and bottom edges.

The island is partitioned into a grid of square cells. You are given an m x n integer matrix heights where heights[r][c] represents the height above sea level of the cell at coordinate (r, c).

The island receives a lot of rain, and the rain water can flow to neighboring cells directly north, south, east, and west if the neighboring cell's height is less than or equal to the current cell's height. Water can flow from any cell adjacent to an ocean into the ocean.

Return a 2D list of grid coordinates result where result[i] = [ri, ci] denotes that rain water can flow from cell (ri, ci) to both the Pacific and Atlantic oceans.

 

Example 1:

Input: heights = [[1,2,2,3,5],[3,2,3,4,4],[2,4,5,3,1],[6,7,1,4,5],[5,1,1,2,4]]
Output: [[0,4],[1,3],[1,4],[2,2],[3,0],[3,1],[4,0]]
Explanation: The following cells can flow to the Pacific and Atlantic oceans, as shown below:
[0,4]: [0,4] -> Pacific Ocean
      [0,4] -> Atlantic Ocean
[1,3]: [1,3] -> Pacific Ocean
      [1,3] -> Atlantic Ocean
[1,4]: [1,4] -> Pacific Ocean
      [1,4] -> Atlantic Ocean
[2,2]: [2,2] -> Pacific Ocean
      [2,2] -> Atlantic Ocean
[3,0]: [3,0] -> Pacific Ocean
      [3,0] -> Atlantic Ocean
[3,1]: [3,1] -> Pacific Ocean
      [3,1] -> Atlantic Ocean
[4,0]: [4,0] -> Pacific Ocean
      [4,0] -> Atlantic Ocean

Example 2:

Input: heights = [[1]]
Output: [[0,0]]
Explanation: The water can flow from the only cell to the Pacific and Atlantic oceans.

 

Constraints:

    m == heights.length
    n == heights[i].length
    1 <= m, n <= 200
    0 <= heights[i][j] <= 105

https://leetcode.com/problems/pacific-atlantic-water-flow/description/
"""

        def easy_solution(heights: List[List[int]]) -> List[List[int]]:
            # Easy solution: DFS from both oceans to find reachable cells
            if not heights or not heights[0]:
                return []

            m, n = len(heights), len(heights[0])
            pacific = set()
            atlantic = set()

            def dfs(i: int, j: int, reachable: set) -> None:
                reachable.add((i, j))
                for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    ni, nj = i + di, j + dj
                    if (
                        0 <= ni < m
                        and 0 <= nj < n
                        and (ni, nj) not in reachable
                        and heights[ni][nj] >= heights[i][j]
                    ):
                        dfs(ni, nj, reachable)

            for i in range(m):
                dfs(i, 0, pacific)
                dfs(i, n - 1, atlantic)

            for j in range(n):
                dfs(0, j, pacific)
                dfs(m - 1, j, atlantic)

            return list(pacific & atlantic)

        def optimized_solution(heights: List[List[int]]) -> List[List[int]]:
            # Optimized solution: Use boolean arrays to track reachable cells
            if not heights or not heights[0]:
                return []

            m, n = len(heights), len(heights[0])
            pacific = [[False] * n for _ in range(m)]
            atlantic = [[False] * n for _ in range(m)]

            def dfs(i: int, j: int, reachable: List[List[bool]]) -> None:
                reachable[i][j] = True
                for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    ni, nj = i + di, j + dj
                    if (
                        0 <= ni < m
                        and 0 <= nj < n
                        and not reachable[ni][nj]
                        and heights[ni][nj] >= heights[i][j]
                    ):
                        dfs(ni, nj, reachable)

            for i in range(m):
                dfs(i, 0, pacific)
                dfs(i, n - 1, atlantic)

            for j in range(n):
                dfs(0, j, pacific)
                dfs(m - 1, j, atlantic)

            return [
                [i, j]
                for i in range(m)
                for j in range(n)
                if pacific[i][j] and atlantic[i][j]
            ]

        return Solution(
            question="Pacific Atlantic Water Flow",
            easy_solution=easy_solution,
            optimized_solution=optimized_solution,
            time_complexity="O(m * n)",
            space_complexity="O(m * n)",
            similar_questions=["Number of Islands", "Surrounded Regions"],
            problem_statement=problem_statement,
        )

    @staticmethod
    def number_of_islands() -> Solution:
        """
        Generate a Solution object for the Number of Islands problem.

        Returns:
            Solution: A Solution object containing details of the Number of Islands problem.
        """

        def problem_statement() -> str:
            return """Given an m x n 2D binary grid grid which represents a map of '1's (land) and '0's (water), return the number of islands.

An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.

 

Example 1:

Input: grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
Output: 1

Example 2:

Input: grid = [
  ["1","0","0","0","0"],
  ["0","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","0"]
]
Output: 4

 

Constraints:

    m == grid.length
    n == grid[i].length
    1 <= m, n <= 300
    grid[i][j] is '0' or '1'.

https://leetcode.com/problems/number-of-islands/description/
"""

        def easy_solution(grid: List[List[str]]) -> int:
            # Easy solution: DFS to count islands
            if not grid or not grid[0]:
                return 0

            m, n = len(grid), len(grid[0])
            islands = 0

            def dfs(i: int, j: int) -> None:
                if i < 0 or i >= m or j < 0 or j >= n or grid[i][j] == "0":
                    return
                grid[i][j] = "0"
                dfs(i + 1, j)
                dfs(i - 1, j)
                dfs(i, j + 1)
                dfs(i, j - 1)

            for i in range(m):
                for j in range(n):
                    if grid[i][j] == "1":
                        islands += 1
                        dfs(i, j)

            return islands

        def optimized_solution(grid: List[List[str]]) -> int:
            # Optimized solution: BFS to count islands
            if not grid or not grid[0]:
                return 0

            m, n = len(grid), len(grid[0])
            islands = 0

            def bfs(i: int, j: int) -> None:
                queue = deque([(i, j)])
                while queue:
                    i, j = queue.popleft()
                    for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < m and 0 <= nj < n and grid[ni][nj] == "1":
                            grid[ni][nj] = "0"
                            queue.append((ni, nj))

            for i in range(m):
                for j in range(n):
                    if grid[i][j] == "1":
                        islands += 1
                        grid[i][j] = "0"
                        bfs(i, j)

            return islands

        return Solution(
            question="Number of Islands",
            easy_solution=easy_solution,
            optimized_solution=optimized_solution,
            time_complexity="O(m * n)",
            space_complexity="O(min(m, n))",
            similar_questions=[
                "Surrounded Regions",
                "Walls and Gates",
                "Number of Islands II",
                "Number of Distinct Islands",
                "Max Area of Island",
            ],
            problem_statement=problem_statement,
        )

    @staticmethod
    def longest_consecutive_sequence() -> Solution:
        """
        Generate a Solution object for the Longest Consecutive Sequence problem.

        Returns:
            Solution: A Solution object containing details of the Longest Consecutive Sequence problem.
        """

        def problem_statement() -> str:
            return """Given an unsorted array of integers nums, return the length of the longest consecutive elements sequence.

You must write an algorithm that runs in O(n) time.

 

Example 1:

Input: nums = [100,4,200,1,3,2]
Output: 4
Explanation: The longest consecutive elements sequence is [1, 2, 3, 4]. Therefore its length is 4.

Example 2:

Input: nums = [0,3,7,2,5,8,4,6,0,1]
Output: 9

 

Constraints:

    0 <= nums.length <= 105
    -109 <= nums[i] <= 109

https://leetcode.com/problems/longest-consecutive-sequence/description/
"""

        def easy_solution(nums: List[int]) -> int:
            # Easy solution: Use a set to find the longest consecutive sequence
            if not nums:
                return 0

            num_set = set(nums)
            longest = 0

            for num in num_set:
                if num - 1 not in num_set:
                    current = num
                    streak = 1

                    while current + 1 in num_set:
                        current += 1
                        streak += 1

                    longest = max(longest, streak)

            return longest

        def optimized_solution(nums: List[int]) -> int:
            # Optimized solution: Similar approach but streamlined
            num_set = set(nums)
            longest = 0

            for num in nums:
                if num - 1 not in num_set:
                    current = num
                    streak = 1

                    while current + 1 in num_set:
                        current += 1
                        streak += 1

                    longest = max(longest, streak)

            return longest

        return Solution(
            question="Longest Consecutive Sequence",
            easy_solution=easy_solution,
            optimized_solution=optimized_solution,
            time_complexity="O(n)",
            space_complexity="O(n)",
            similar_questions=["Binary Tree Longest Consecutive Sequence"],
            problem_statement=problem_statement,
        )

    @staticmethod
    def alien_dictionary() -> Solution:
        """
        Generate a Solution object for the Alien Dictionary problem.

        Returns:
            Solution: A Solution object containing details of the Alien Dictionary problem.
        """

        def problem_statement() -> str:
            return """There is a new alien language that uses the English alphabet. However, the order among letters are unknown to you.

You are given a list of strings words from the alien language's dictionary, where the strings in words are sorted lexicographically by the rules of this new language.

Return a string of the unique letters in the new alien language sorted in lexicographically increasing order by the new language's rules. If there is no solution, return "". If there are multiple solutions, return any of them.

A string s is lexicographically smaller than a string t if at the first letter where they differ, the letter in s comes before the letter in t in the alien language. If the first min(s.length, t.length) letters are the same, then s is smaller if and only if s.length < t.length.

 

Example 1:

Input: words = ["wrt","wrf","er","ett","rftt"]
Output: "wertf"

Example 2:

Input: words = ["z","x"]
Output: "zx"

Example 3:

Input: words = ["z","x","z"]
Output: ""
Explanation: The order is invalid, so return "".

 

Constraints:

    1 <= words.length <= 100
    1 <= words[i].length <= 100
    words[i] consists of only lowercase English letters.

https://leetcode.com/problems/alien-dictionary/description/
"""

        def easy_solution(words: List[str]) -> str:
            # Easy solution: DFS to build the order
            graph: Dict[str, set] = {c: set() for word in words for c in word}

            for i in range(len(words) - 1):
                w1, w2 = words[i], words[i + 1]
                min_len = min(len(w1), len(w2))
                if len(w1) > len(w2) and w1[:min_len] == w2[:min_len]:
                    return ""
                for j in range(min_len):
                    if w1[j] != w2[j]:
                        graph[w1[j]].add(w2[j])
                        break

            visited = {}
            result = []

            def dfs(c: str) -> bool:
                if c in visited:
                    return visited[c]
                visited[c] = True
                for nei in graph[c]:
                    if dfs(nei):
                        return True
                visited[c] = False
                result.append(c)
                return False

            for c in graph:
                if dfs(c):
                    return ""

            return "".join(result[::-1])

        def optimized_solution(words: List[str]) -> str:
            # Optimized solution: BFS with in-degree tracking
            graph: Dict[str, set] = {c: set() for word in words for c in word}
            in_degree: Dict[str, int] = {c: 0 for word in words for c in word}

            for i in range(len(words) - 1):
                w1, w2 = words[i], words[i + 1]
                min_len = min(len(w1), len(w2))
                if len(w1) > len(w2) and w1[:min_len] == w2[:min_len]:
                    return ""
                for j in range(min_len):
                    if w1[j] != w2[j]:
                        if w2[j] not in graph[w1[j]]:
                            graph[w1[j]].add(w2[j])
                            in_degree[w2[j]] += 1
                        break

            queue = deque([c for c in in_degree if in_degree[c] == 0])
            result = []

            while queue:
                c = queue.popleft()
                result.append(c)
                for nei in graph[c]:
                    in_degree[nei] -= 1
                    if in_degree[nei] == 0:
                        queue.append(nei)

            if len(result) != len(graph):
                return ""
            return "".join(result)

        return Solution(
            question="Alien Dictionary",
            easy_solution=easy_solution,
            optimized_solution=optimized_solution,
            time_complexity="O(C)",
            space_complexity="O(1)",
            similar_questions=["Course Schedule II", "Sequence Reconstruction"],
            problem_statement=problem_statement,
        )

    @staticmethod
    def graph_valid_tree() -> Solution:
        """
        Generate a Solution object for the Graph Valid Tree problem.

        Returns:
            Solution: A Solution object containing details of the Graph Valid Tree problem.
        """

        def problem_statement() -> str:
            return """You have a graph of n nodes labeled from 0 to n - 1. You are given an integer n and a list of edges where edges[i] = [ai, bi] indicates that there is an undirected edge between nodes ai and bi in the graph.

Return true if the edges of the given graph make up a valid tree, and false otherwise.

 

Example 1:

Input: n = 5, edges = [[0,1],[0,2],[0,3],[1,4]]
Output: true

Example 2:

Input: n = 5, edges = [[0,1],[1,2],[2,3],[1,3],[1,4]]
Output: false

 

Constraints:

    1 <= n <= 2000
    0 <= edges.length <= 5000
    edges[i].length == 2
    0 <= ai, bi < n
    ai != bi
    There are no self-loops or repeated edges.

https://leetcode.com/problems/graph-valid-tree/description/
"""

        def easy_solution(n: int, edges: List[List[int]]) -> bool:
            # Easy solution: DFS to check for cycles and connectivity
            if len(edges) != n - 1:
                return False

            graph = [[] for _ in range(n)]
            for u, v in edges:
                graph[u].append(v)
                graph[v].append(u)

            visited = set()

            def dfs(node: int, parent: int) -> bool:
                visited.add(node)
                for neighbor in graph[node]:
                    if neighbor == parent:
                        continue
                    if neighbor in visited:
                        return False
                    if not dfs(neighbor, node):
                        return False
                return True

            return dfs(0, -1) and len(visited) == n

        def optimized_solution(n: int, edges: List[List[int]]) -> bool:
            # Optimized solution: Union-Find to check for cycles and connectivity
            if len(edges) != n - 1:
                return False

            parent = list(range(n))

            def find(x: int) -> int:
                if parent[x] != x:
                    parent[x] = find(parent[x])
                return parent[x]

            def union(x: int, y: int) -> bool:
                root_x, root_y = find(x), find(y)
                if root_x == root_y:
                    return False
                parent[root_x] = root_y
                return True

            return all(union(u, v) for u, v in edges)

        return Solution(
            question="Graph Valid Tree",
            easy_solution=easy_solution,
            optimized_solution=optimized_solution,
            time_complexity="O(n)",
            space_complexity="O(n)",
            similar_questions=[
                "Number of Connected Components in an Undirected Graph",
                "Redundant Connection",
            ],
            problem_statement=problem_statement,
        )

    @staticmethod
    def number_of_connected_components() -> Solution:
        """
        Generate a Solution object for the Number of Connected Components in an Undirected Graph problem.

        Returns:
            Solution: A Solution object containing details of the Number of Connected Components in an Undirected Graph problem.
        """

        def problem_statement() -> str:
            return """You have a graph of n nodes. You are given an integer n and an array edges where edges[i] = [ai, bi] indicates that there is an edge between ai and bi in the graph.

Return the number of connected components in the graph.

 

Example 1:

Input: n = 5, edges = [[0,1],[1,2],[3,4]]
Output: 2

Example 2:

Input: n = 5, edges = [[0,1],[1,2],[2,3],[3,4]]
Output: 1

 

Constraints:

    1 <= n <= 2000
    0 <= edges.length <= 5000
    edges[i].length == 2
    0 <= ai, bi < n
    ai != bi
    There are no repeated edges.

https://leetcode.com/problems/number-of-connected-components-in-an-undirected-graph/description/
"""

        def easy_solution(n: int, edges: List[List[int]]) -> int:
            # Easy solution: DFS to count connected components
            graph = [[] for _ in range(n)]
            for u, v in edges:
                graph[u].append(v)
                graph[v].append(u)

            visited = set()
            components = 0

            def dfs(node: int) -> None:
                visited.add(node)
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        dfs(neighbor)

            for node in range(n):
                if node not in visited:
                    dfs(node)
                    components += 1

            return components

        def optimized_solution(n: int, edges: List[List[int]]) -> int:
            # Optimized solution: Union-Find to count connected components
            parent = list(range(n))
            rank = [0] * n
            components = n

            def find(x: int) -> int:
                if parent[x] != x:
                    parent[x] = find(parent[x])
                return parent[x]

            def union(x: int, y: int) -> None:
                nonlocal components
                root_x, root_y = find(x), find(y)
                if root_x != root_y:
                    if rank[root_x] < rank[root_y]:
                        parent[root_x] = root_y
                    elif rank[root_x] > rank[root_y]:
                        parent[root_y] = root_x
                    else:
                        parent[root_y] = root_x
                        rank[root_x] += 1
                    components -= 1

            for u, v in edges:
                union(u, v)

            return components

        return Solution(
            question="Number of Connected Components in an Undirected Graph",
            easy_solution=easy_solution,
            optimized_solution=optimized_solution,
            time_complexity="O(n)",
            space_complexity="O(n)",
            similar_questions=[
                "Number of Islands",
                "Graph Valid Tree",
                "Friend Circles",
            ],
            problem_statement=problem_statement,
        )
