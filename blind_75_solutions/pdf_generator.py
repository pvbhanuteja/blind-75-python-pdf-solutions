"""
Module for generating PDF content for Blind 75 solutions.
"""

from typing import List
from markdown_pdf import MarkdownPdf, Section
from blind_75_solutions import Solution


class PdfGenerator:
    """
    A class to generate PDF content for Blind 75 solutions.

    Attributes:
        solutions (List[Solution]): A list of Solution objects.
    """

    def __init__(self, solutions: List[Solution]) -> None:
        """
        Initialize a PdfGenerator object.

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
            toc += f"{index}. {solution.question}\n"
        return toc

    def generate_full_pdf(self, output_file: str) -> None:
        """
        Generate the full PDF content for all solutions.

        Args:
            output_file (str): The path to save the output PDF file.
        """
        pdf = MarkdownPdf(toc_level=2)
        
        # Add title and table of contents
        pdf.add_section(Section("# Blind 75 LeetCode Solutions\n\n", toc=False))
        pdf.add_section(Section(self.generate_table_of_contents()))

        # Add each solution
        for solution in self.solutions:
            pdf.add_section(Section(solution.generate_markdown()))

        # Set PDF metadata
        pdf.meta["title"] = "Blind 75 LeetCode Solutions"
        pdf.meta["author"] = "Your Name"
        pdf.meta["subject"] = "LeetCode Solutions"
        pdf.meta["keywords"] = "leetcode, algorithms, data structures"

        # Save the PDF
        pdf.save(output_file)