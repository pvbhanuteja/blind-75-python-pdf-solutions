"""
Module containing matrix-related solutions for Blind 75 LeetCode questions.
"""

from typing import List
from blind_75_solutions import Solution


class MatrixSolutions:
    """Class containing matrix-related solutions for Blind 75 LeetCode questions."""

    @staticmethod
    def set_matrix_zeroes() -> Solution:
        """
        Generate a Solution object for the Set Matrix Zeroes problem.

        Returns:
            Solution: A Solution object containing details of the Set Matrix Zeroes problem.
        """

        def problem_statement() -> str:
            return """Given an m x n integer matrix matrix, if an element is 0, set its entire row and column to 0's.

You must do it in place.

 

Example 1:

Input: matrix = [[1,1,1],[1,0,1],[1,1,1]]
Output: [[1,0,1],[0,0,0],[1,0,1]]

Example 2:

Input: matrix = [[0,1,2,0],[3,4,5,2],[1,3,1,5]]
Output: [[0,0,0,0],[0,4,5,0],[0,3,1,0]]

 

Constraints:

    m == matrix.length
    n == matrix[0].length
    1 <= m, n <= 200
    -231 <= matrix[i][j] <= 231 - 1

https://leetcode.com/problems/set-matrix-zeroes/description/
"""

        def easy_solution(matrix: List[List[int]]) -> None:
            # Brute-force solution: Use sets to track zero rows and columns
            m, n = len(matrix), len(matrix[0])
            zero_rows, zero_cols = set(), set()

            for i in range(m):
                for j in range(n):
                    if matrix[i][j] == 0:
                        zero_rows.add(i)
                        zero_cols.add(j)

            for i in range(m):
                for j in range(n):
                    if i in zero_rows or j in zero_cols:
                        matrix[i][j] = 0

        def optimized_solution(matrix: List[List[int]]) -> None:
            # Optimized solution: Use first row and column as markers
            m, n = len(matrix), len(matrix[0])
            first_row_zero = False
            first_col_zero = False

            # Check if the first row has any zeros
            for j in range(n):
                if matrix[0][j] == 0:
                    first_row_zero = True
                    break

            # Check if the first column has any zeros
            for i in range(m):
                if matrix[i][0] == 0:
                    first_col_zero = True
                    break

            # Use first row and column to mark zeros
            for i in range(1, m):
                for j in range(1, n):
                    if matrix[i][j] == 0:
                        matrix[i][0] = 0
                        matrix[0][j] = 0

            # Set matrix elements to zero based on markers
            for i in range(1, m):
                for j in range(1, n):
                    if matrix[i][0] == 0 or matrix[0][j] == 0:
                        matrix[i][j] = 0

            # Set the first row to zero if needed
            if first_row_zero:
                for j in range(n):
                    matrix[0][j] = 0

            # Set the first column to zero if needed
            if first_col_zero:
                for i in range(m):
                    matrix[i][0] = 0

        return Solution(
            question="Set Matrix Zeroes",
            easy_solution=easy_solution,
            optimized_solution=optimized_solution,
            time_complexity="O(m * n)",
            space_complexity="O(1)",
            similar_questions=["Game of Life"],
            problem_statement=problem_statement,
        )

    @staticmethod
    def spiral_matrix() -> Solution:
        """
        Generate a Solution object for the Spiral Matrix problem.

        Returns:
            Solution: A Solution object containing details of the Spiral Matrix problem.
        """

        def problem_statement() -> str:
            return """Given an m x n matrix, return all elements of the matrix in spiral order.

 

Example 1:

Input: matrix = [[1,2,3],[4,5,6],[7,8,9]]
Output: [1,2,3,6,9,8,7,4,5]

Example 2:

Input: matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
Output: [1,2,3,4,8,12,11,10,9,5,6,7]

 

Constraints:

    m == matrix.length
    n == matrix[0].length
    1 <= m, n <= 10
    -100 <= matrix[i][j] <= 100

https://leetcode.com/problems/spiral-matrix/description/
"""

        def easy_solution(matrix: List[List[int]]) -> List[int]:
            # Brute-force solution: Traverse the matrix in spiral order
            if not matrix:
                return []

            result = []
            top, bottom, left, right = 0, len(matrix) - 1, 0, len(matrix[0]) - 1

            while top <= bottom and left <= right:
                for j in range(left, right + 1):
                    result.append(matrix[top][j])
                top += 1

                for i in range(top, bottom + 1):
                    result.append(matrix[i][right])
                right -= 1

                if top <= bottom:
                    for j in range(right, left - 1, -1):
                        result.append(matrix[bottom][j])
                    bottom -= 1

                if left <= right:
                    for i in range(bottom, top - 1, -1):
                        result.append(matrix[i][left])
                    left += 1

            return result

        def optimized_solution(matrix: List[List[int]]) -> List[int]:
            # Optimized solution: Use direction vectors to traverse the matrix
            if not matrix:
                return []

            result = []
            m, n = len(matrix), len(matrix[0])
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            d = 0
            row, col = 0, 0
            visited = set()

            for _ in range(m * n):
                result.append(matrix[row][col])
                visited.add((row, col))

                next_row, next_col = row + directions[d][0], col + directions[d][1]

                if (
                    next_row < 0
                    or next_row >= m
                    or next_col < 0
                    or next_col >= n
                    or (next_row, next_col) in visited
                ):
                    d = (d + 1) % 4
                    next_row, next_col = row + directions[d][0], col + directions[d][1]

                row, col = next_row, next_col

            return result

        return Solution(
            question="Spiral Matrix",
            easy_solution=easy_solution,
            optimized_solution=optimized_solution,
            time_complexity="O(m * n)",
            space_complexity="O(1)",
            similar_questions=["Spiral Matrix II", "Spiral Matrix III"],
            problem_statement=problem_statement,
        )

    @staticmethod
    def rotate_image() -> Solution:
        """
        Generate a Solution object for the Rotate Image problem.

        Returns:
            Solution: A Solution object containing details of the Rotate Image problem.
        """

        def problem_statement() -> str:
            return """You are given an n x n 2D matrix representing an image, rotate the image by 90 degrees (clockwise).

You have to rotate the image in-place, which means you have to modify the input 2D matrix directly. DO NOT allocate another 2D matrix and do the rotation.

 

Example 1:

Input: matrix = [[1,2,3],[4,5,6],[7,8,9]]
Output: [[7,4,1],[8,5,2],[9,6,3]]

Example 2:

Input: matrix = [[5,1,9,11],[2,4,8,10],[13,3,6,7],[15,14,12,16]]
Output: [[15,13,2,5],[14,3,4,1],[12,6,8,9],[16,7,10,11]]

 

Constraints:

    n == matrix.length == matrix[i].length
    1 <= n <= 20
    -1000 <= matrix[i][j] <= 1000

https://leetcode.com/problems/rotate-image/description/
"""

        def easy_solution(matrix: List[List[int]]) -> None:
            # Brute-force solution: Transpose the matrix, then reverse each row
            n = len(matrix)

            # Transpose the matrix
            for i in range(n):
                for j in range(i, n):
                    matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

            # Reverse each row
            for i in range(n):
                matrix[i].reverse()

        def optimized_solution(matrix: List[List[int]]) -> None:
            # Optimized solution: Rotate four rectangles
            n = len(matrix)

            # Rotate four rectangles
            for i in range(n // 2 + n % 2):
                for j in range(n // 2):
                    tmp = matrix[n - 1 - j][i]
                    matrix[n - 1 - j][i] = matrix[n - 1 - i][n - j - 1]
                    matrix[n - 1 - i][n - j - 1] = matrix[j][n - 1 - i]
                    matrix[j][n - 1 - i] = matrix[i][j]
                    matrix[i][j] = tmp

        return Solution(
            question="Rotate Image",
            easy_solution=easy_solution,
            optimized_solution=optimized_solution,
            time_complexity="O(n^2)",
            space_complexity="O(1)",
            similar_questions=["Determine Whether Matrix Can Be Obtained By Rotation"],
            problem_statement=problem_statement,
        )

    @staticmethod
    def word_search() -> Solution:
        """
        Generate a Solution object for the Word Search problem.

        Returns:
            Solution: A Solution object containing details of the Word Search problem.
        """

        def problem_statement() -> str:
            return """Given an m x n grid of characters board and a string word, return true if word exists in the grid.

The word can be constructed from letters of sequentially adjacent cells, where adjacent cells are horizontally or vertically neighboring. The same letter cell may not be used more than once.

 

Example 1:

Input: board = [['A','B','C','E'],['S','F','C','S'],['A','D','E','E']], word = "ABCCED"
Output: true

Example 2:

Input: board = [['A','B','C','E'],['S','F','C','S'],['A','D','E','E']], word = "SEE"
Output: true

Example 3:

Input: board = [['A','B','C','E'],['S','F','C','S'],['A','D','E','E']], word = "ABCB"
Output: false

 

Constraints:

    m == board.length
    n == board[i].length
    1 <= m, n <= 6
    1 <= word.length <= 15
    board and word consists of only lowercase and uppercase English letters.

https://leetcode.com/problems/word-search/description/
"""

        def easy_solution(board: List[List[str]], word: str) -> bool:
            # Brute-force solution: Use DFS to check for the word
            def dfs(i: int, j: int, k: int) -> bool:
                if k == len(word):
                    return True
                if (
                    i < 0
                    or i >= len(board)
                    or j < 0
                    or j >= len(board[0])
                    or board[i][j] != word[k]
                ):
                    return False

                temp, board[i][j] = board[i][j], "#"
                result = (
                    dfs(i + 1, j, k + 1)
                    or dfs(i - 1, j, k + 1)
                    or dfs(i, j + 1, k + 1)
                    or dfs(i, j - 1, k + 1)
                )
                board[i][j] = temp
                return result

            for i in range(len(board)):
                for j in range(len(board[0])):
                    if dfs(i, j, 0):
                        return True
            return False

        def optimized_solution(board: List[List[str]], word: str) -> bool:
            # Optimized solution: Use DFS with visited array
            def dfs(i: int, j: int, k: int) -> bool:
                if k == len(word):
                    return True
                if (
                    i < 0
                    or i >= m
                    or j < 0
                    or j >= n
                    or visited[i][j]
                    or board[i][j] != word[k]
                ):
                    return False

                visited[i][j] = True
                for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    if dfs(i + di, j + dj, k + 1):
                        return True
                visited[i][j] = False
                return False

            m, n = len(board), len(board[0])
            visited = [[False] * n for _ in range(m)]

            for i in range(m):
                for j in range(n):
                    if board[i][j] == word[0] and dfs(i, j, 0):
                        return True
            return False

        return Solution(
            question="Word Search",
            easy_solution=easy_solution,
            optimized_solution=optimized_solution,
            time_complexity="O(m * n * 4^L)",
            space_complexity="O(m * n)",
            similar_questions=["Word Search II"],
            problem_statement=problem_statement,
        )
