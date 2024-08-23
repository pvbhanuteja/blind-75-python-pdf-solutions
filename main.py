## main.py

"""
Main module for generating Blind 75 LeetCode solutions markdown.
"""

import os
from typing import List
from blind_75_solutions.markdown_generator import MarkdownGenerator
from blind_75_solutions.array_solutions import ArraySolutions
from blind_75_solutions.string_solutions import StringSolutions
from blind_75_solutions.linked_list_solutions import LinkedListSolutions
from blind_75_solutions.tree_solutions import TreeSolutions
from blind_75_solutions.dynamic_programming_solutions import DynamicProgrammingSolutions
from blind_75_solutions.graph_solutions import GraphSolutions
from blind_75_solutions.interval_solutions import IntervalSolutions
from blind_75_solutions.matrix_solutions import MatrixSolutions
from blind_75_solutions.heap_solutions import HeapSolutions
from blind_75_solutions.binary_solutions import BinarySolutions
from blind_75_solutions import Solution
from blind_75_solutions.pdf_generator import PdfGenerator


def get_all_solutions() -> List[Solution]:
    """
    Collect all solutions from different categories.

    Returns:
        List[Solution]: A list of all Solution objects.
    """
    solution_classes = [
        ArraySolutions(),
        StringSolutions(),
        BinarySolutions(),
        LinkedListSolutions(),
        TreeSolutions(),
        DynamicProgrammingSolutions(),
        GraphSolutions(),
        IntervalSolutions(),
        MatrixSolutions(),
        HeapSolutions(),
    ]

    all_solutions = []
    for solution_class in solution_classes:
        for method_name in reversed(dir(solution_class)):
            if not method_name.startswith("__"):
                method = getattr(solution_class, method_name)
                if callable(method):
                    all_solutions.append(method())

    return all_solutions


def main() -> None:
    """
    Main function to generate the Blind 75 LeetCode solutions markdown.
    """
    all_solutions = get_all_solutions()
    markdown_generator = MarkdownGenerator(all_solutions)
    markdown_content = markdown_generator.generate_full_markdown()

    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, "blind_75_solutions.md")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    print(f"Markdown file generated successfully: {output_file}")

    # Generate PDF file
    pdf_generator = PdfGenerator(all_solutions)
    # output_pdf_file = os.path.join(output_dir, "blind_75_solutions_old.pdf")
    # pdf_generator.generate_full_pdf(output_pdf_file)
    fpdf_output_file = os.path.join(output_dir, "blind_75_solutions.pdf")
    pdf_generator.generate_full_pdf_fpdf(fpdf_output_file)

    print(f"PDF file generated successfully: {fpdf_output_file}")


if __name__ == "__main__":
    main()
