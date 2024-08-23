## blind_75_solutions/markdown_generator.py

"""
Module for generating markdown content for Blind 75 solutions.
"""

from typing import List
from blind_75_solutions import Solution


class MarkdownGenerator:
    """
    A class to generate markdown content for Blind 75 solutions.

    Attributes:
        solutions (List[Solution]): A list of Solution objects.
    """

    def __init__(self, solutions: List[Solution]) -> None:
        """
        Initialize a MarkdownGenerator object.

        Args:
            solutions (List[Solution]): A list of Solution objects.
        """
        self.solutions: List[Solution] = solutions

    def generate_table_of_contents(self) -> str:
        """
        Generate a table of contents for the solutions.

        Returns:
            str: A markdown string containing the table of contents.
        """
        toc = "# Table of Contents\n\n"
        for index, solution in enumerate(self.solutions, start=1):
            toc += f"{index}. [{solution.question}](#{solution.question.lower().replace(' ', '-')})\n"
        return toc

    def generate_full_markdown(self) -> str:
        """
        Generate the full markdown content for all solutions.

        Returns:
            str: A markdown string containing all solution details.
        """
        markdown = "# Blind 75 LeetCode Solutions\n\n"
        markdown += self.generate_table_of_contents()
        markdown += "\n"

        for solution in self.solutions:
            markdown += solution.generate_markdown()
            markdown += "\n---\n\n"

        return markdown
