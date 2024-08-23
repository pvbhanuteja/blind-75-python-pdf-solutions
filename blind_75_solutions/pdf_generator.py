"""
Module for generating PDF content for Blind 75 solutions.
"""

from typing import List
from markdown_pdf import MarkdownPdf, Section
from blind_75_solutions import Solution
from mistletoe import markdown, Document
from mistletoe.html_renderer import HTMLRenderer
from fpdf import FPDF, HTMLMixin
import html


class CodeHTMLRenderer(HTMLRenderer):
    def escape_html(self, text):
        return html.escape(text)

    def render_block_code(self, token):
        code = self.escape_html(token.children[0].content)
        lines = code.split("\n")
        formatted_code = "<br>".join(
            f"&nbsp;&nbsp;&nbsp;&nbsp;{line}" for line in lines
        )
        return f'<pre style="background-color: #f0f0f0; padding: 10px; font-family: Courier, monospace; font-size: 10px; white-space: pre-wrap; word-wrap: break-word;">{formatted_code}</pre>'


class MyFPDF(FPDF, HTMLMixin):
    pass


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

    def generate_full_pdf_fpdf(self, output_file: str) -> None:
        pdf = MyFPDF(orientation="P", unit="mm", format="A4")
        pdf.add_page()
        pdf.add_font("DejaVuSans", fname="./fonts/DejaVuSans.ttf", uni=True)
        pdf.add_font(
            "DejaVuSans", fname="./fonts/DejaVuSans-Bold.ttf", style="B", uni=True
        )
        pdf.set_font("DejaVuSans", size=12)

        renderer = CodeHTMLRenderer()

        # Title and Table of Contents
        content = (
            f"# Blind 75 LeetCode Solutions\n\n{self.generate_table_of_contents()}"
        )
        pdf.write_html(renderer.render(Document(content)))

        # Add each solution
        for solution in self.solutions:
            pdf.add_page()
            pdf.write_html(renderer.render(Document(solution.generate_markdown())))

        pdf.output(output_file)
