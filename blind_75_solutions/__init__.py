## blind_75_solutions/__init__.py

"""
Blind 75 Solutions package initialization.

This package contains solutions for the Blind 75 LeetCode questions,
organized by problem category.
"""

from typing import Callable, List
import inspect

class Solution:
    """
    A class to represent a solution for a Blind 75 problem.

    Attributes:
        question (str): The problem statement.
        easy_solution (Callable): A function implementing an easy solution.
        optimized_solution (Callable): A function implementing an optimized solution.
        time_complexity (str): The time complexity of the optimized solution.
        space_complexity (str): The space complexity of the optimized solution.
        similar_questions (List[str]): A list of similar interview questions.
    """

    def __init__(
        self,
        question: str,
        problem_statement: Callable,
        easy_solution: Callable,
        optimized_solution: Callable,
        time_complexity: str,
        space_complexity: str,
        similar_questions: List[str]
    ) -> None:
        """
        Initialize a Solution object.

        Args:
            question (str): The problem statement.
            easy_solution (Callable): A function implementing an easy solution.
            optimized_solution (Callable): A function implementing an optimized solution.
            time_complexity (str): The time complexity of the optimized solution.
            space_complexity (str): The space complexity of the optimized solution.
            similar_questions (List[str]): A list of similar interview questions.
        """
        self.question = question
        self.problem_statement = problem_statement
        self.easy_solution = easy_solution
        self.optimized_solution = optimized_solution
        self.time_complexity = time_complexity
        self.space_complexity = space_complexity
        self.similar_questions = similar_questions

    def generate_markdown(self) -> str:
        """
        Generate a markdown representation of the solution.

        Returns:
            str: A markdown string containing the solution details.
        """
        markdown = f"# {self.question}\n\n"
        markdown += f"{self.problem_statement()}\n\n"
        markdown += "## Easy Solution\n"
        markdown += f"```python\n{inspect.getsource(self.easy_solution)}\n```\n\n"
        markdown += "## Optimized Solution\n"
        markdown += f"```python\n{inspect.getsource(self.optimized_solution)}\n```\n\n"
        markdown += f"Time Complexity: {self.time_complexity}\n"
        markdown += f"Space Complexity: {self.space_complexity}\n\n"
        markdown += "## Similar Questions\n"
        for question in self.similar_questions:
            markdown += f"- {question}\n"
        return markdown