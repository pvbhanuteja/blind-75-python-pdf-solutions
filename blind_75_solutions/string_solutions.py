"""
Module containing string-related solutions for Blind 75 LeetCode questions.
"""

from typing import List, Dict
from blind_75_solutions import Solution


class StringSolutions:
    """Class containing string-related solutions for Blind 75 LeetCode questions."""

    @staticmethod
    def longest_substring_without_repeating_characters() -> Solution:
        """
        Generate a Solution object for the Longest Substring Without Repeating Characters problem.

        Returns:
            Solution: A Solution object containing details of the problem.
        """
        def problem_statement() -> str:
            return '''Given a string s, find the length of the longest substring without repeating characters.

Example 1:
Input: s = "abcabcbb"
Output: 3
Explanation: The answer is "abc", with the length of 3.

Example 2:
Input: s = "bbbbb"
Output: 1
Explanation: The answer is "b", with the length of 1.

Example 3:
Input: s = "pwwkew"
Output: 3
Explanation: The answer is "wke", with the length of 3.
Notice that the answer must be a substring, "pwke" is a subsequence and not a substring.

Constraints:
    0 <= s.length <= 5 * 104
    s consists of English letters, digits, symbols and spaces.

https://leetcode.com/problems/longest-substring-without-repeating-characters/description/
'''

        def easy_solution(s: str) -> int:
            # Brute-force solution: Check all substrings
            max_length = 0
            for i in range(len(s)):
                seen = set()
                for j in range(i, len(s)):
                    if s[j] in seen:
                        break
                    seen.add(s[j])
                max_length = max(max_length, len(seen))
            return max_length

        def optimized_solution(s: str) -> int:
            # Optimized solution: Use sliding window and hashmap
            char_index = {}
            max_length = start = 0
            for i, char in enumerate(s):
                if char in char_index and start <= char_index[char]:
                    start = char_index[char] + 1
                else:
                    max_length = max(max_length, i - start + 1)
                char_index[char] = i
            return max_length

        return Solution(
            question="Longest Substring Without Repeating Characters",
            easy_solution=easy_solution,
            optimized_solution=optimized_solution,
            time_complexity="O(n)",
            space_complexity="O(min(m, n))",
            similar_questions=["Longest Substring with At Most Two Distinct Characters", "Longest Substring with At Most K Distinct Characters"],
            problem_statement=problem_statement
        )

    @staticmethod
    def longest_repeating_character_replacement() -> Solution:
        """
        Generate a Solution object for the Longest Repeating Character Replacement problem.

        Returns:
            Solution: A Solution object containing details of the problem.
        """
        def problem_statement() -> str:
            return '''You are given a string s and an integer k. You can choose any character of the string and change it to any other uppercase English character. You can perform this operation at most k times.

Return the length of the longest substring containing the same letter you can get after performing the above operations.

Example 1:
Input: s = "ABAB", k = 2
Output: 4
Explanation: Replace the two 'A's with two 'B's or vice versa.

Example 2:
Input: s = "AABABBA", k = 1
Output: 4
Explanation: Replace the one 'A' in the middle with 'B' and form "AABBBBA".

Constraints:
    1 <= s.length <= 105
    s consists of only uppercase English letters.
    0 <= k <= s.length

https://leetcode.com/problems/longest-repeating-character-replacement/description/
'''

        def easy_solution(s: str, k: int) -> int:
            # Brute-force solution: Check all substrings
            max_length = 0
            for i in range(len(s)):
                for j in range(i, len(s)):
                    substring = s[i:j+1]
                    most_common = max(substring.count(c) for c in set(substring))
                    if len(substring) - most_common <= k:
                        max_length = max(max_length, len(substring))
            return max_length

        def optimized_solution(s: str, k: int) -> int:
            # Optimized solution: Use sliding window and hashmap
            char_count = {}
            max_length = max_count = start = 0
            for end, char in enumerate(s):
                char_count[char] = char_count.get(char, 0) + 1
                max_count = max(max_count, char_count[char])
                if end - start + 1 - max_count > k:
                    char_count[s[start]] -= 1
                    start += 1
                max_length = max(max_length, end - start + 1)
            return max_length

        return Solution(
            question="Longest Repeating Character Replacement",
            easy_solution=easy_solution,
            optimized_solution=optimized_solution,
            time_complexity="O(n)",
            space_complexity="O(1)",
            similar_questions=["Longest Substring with At Most K Distinct Characters", "Max Consecutive Ones III"],
            problem_statement=problem_statement
        )

    @staticmethod
    def minimum_window_substring() -> Solution:
        """
        Generate a Solution object for the Minimum Window Substring problem.

        Returns:
            Solution: A Solution object containing details of the problem.
        """
        def problem_statement() -> str:
            return '''Given two strings s and t of lengths m and n respectively, return the minimum window substring of s such that every character in t (including duplicates) is included in the window. If there is no such substring, return the empty string "".

The testcases will be generated such that the answer is unique.

Example 1:
Input: s = "ADOBECODEBANC", t = "ABC"
Output: "BANC"
Explanation: The minimum window substring "BANC" includes 'A', 'B', and 'C' from string t.

Example 2:
Input: s = "a", t = "a"
Output: "a"
Explanation: The entire string s is the minimum window.

Example 3:
Input: s = "a", t = "aa"
Output: ""
Explanation: Both 'a's from t must be included in the window.
Since the largest window of s only has one 'a', return empty string.

Constraints:
    m == s.length
    n == t.length
    1 <= m, n <= 105
    s and t consist of uppercase and lowercase English letters.

Follow up: Could you find an algorithm that runs in O(m + n) time?

https://leetcode.com/problems/minimum-window-substring/description/
'''

        def easy_solution(s: str, t: str) -> str:
            # Brute-force solution: Check all substrings
            def contains_all(window: str, target: str) -> bool:
                return all(window.count(c) >= target.count(c) for c in set(target))

            min_window = ""
            min_length = float('inf')
            for i in range(len(s)):
                for j in range(i, len(s)):
                    window = s[i:j+1]
                    if contains_all(window, t) and len(window) < min_length:
                        min_window = window
                        min_length = len(window)
            return min_window

        def optimized_solution(s: str, t: str) -> str:
            # Optimized solution: Use sliding window and hashmap
            if not t or not s:
                return ""

            dict_t = {}
            for c in t:
                dict_t[c] = dict_t.get(c, 0) + 1

            required = len(dict_t)
            left = right = 0
            formed = 0
            window_counts = {}
            ans = float("inf"), None, None

            while right < len(s):
                character = s[right]
                window_counts[character] = window_counts.get(character, 0) + 1

                if character in dict_t and window_counts[character] == dict_t[character]:
                    formed += 1

                while left <= right and formed == required:
                    character = s[left]

                    if right - left + 1 < ans[0]:
                        ans = (right - left + 1, left, right)

                    window_counts[character] -= 1
                    if character in dict_t and window_counts[character] < dict_t[character]:
                        formed -= 1

                    left += 1    

                right += 1    

            return "" if ans[0] == float("inf") else s[ans[1] : ans[2] + 1]

        return Solution(
            question="Minimum Window Substring",
            easy_solution=easy_solution,
            optimized_solution=optimized_solution,
            time_complexity="O(n)",
            space_complexity="O(k)",
            similar_questions=["Substring with Concatenation of All Words", "Minimum Size Subarray Sum", "Sliding Window Maximum"],
            problem_statement=problem_statement
        )

    @staticmethod
    def valid_anagram() -> Solution:
        """
        Generate a Solution object for the Valid Anagram problem.

        Returns:
            Solution: A Solution object containing details of the problem.
        """
        def problem_statement() -> str:
            return '''Given two strings s and t, return true if t is an anagram of s, and false otherwise.

An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.

Example 1:
Input: s = "anagram", t = "nagaram"
Output: true

Example 2:
Input: s = "rat", t = "car"
Output: false

Constraints:
    1 <= s.length, t.length <= 5 * 104
    s and t consist of lowercase English letters.

Follow up: What if the inputs contain Unicode characters? How would you adapt your solution to such a case?

https://leetcode.com/problems/valid-anagram/description/
'''

        def easy_solution(s: str, t: str) -> bool:
            # Brute-force solution: Sort and compare
            return sorted(s) == sorted(t)

        def optimized_solution(s: str, t: str) -> bool:
            # Optimized solution: Use hashmap to count characters
            if len(s) != len(t):
                return False
            char_count = {}
            for c in s:
                char_count[c] = char_count.get(c, 0) + 1
            for c in t:
                if c not in char_count:
                    return False
                char_count[c] -= 1
                if char_count[c] == 0:
                    del char_count[c]
            return len(char_count) == 0

        return Solution(
            question="Valid Anagram",
            easy_solution=easy_solution,
            optimized_solution=optimized_solution,
            time_complexity="O(n)",
            space_complexity="O(1)",
            similar_questions=["Group Anagrams", "Palindrome Permutation"],
            problem_statement=problem_statement
        )

    @staticmethod
    def group_anagrams() -> Solution:
        """
        Generate a Solution object for the Group Anagrams problem.

        Returns:
            Solution: A Solution object containing details of the problem.
        """
        def problem_statement() -> str:
            return '''Given an array of strings strs, group the anagrams together. You can return the answer in any order.

An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.

Example 1:
Input: strs = ["eat","tea","tan","ate","nat","bat"]
Output: [["bat"],["nat","tan"],["ate","eat","tea"]]

Example 2:
Input: strs = [""]
Output: [[""]]

Example 3:
Input: strs = ["a"]
Output: [["a"]]

Constraints:
    1 <= strs.length <= 104
    0 <= strs[i].length <= 100
    strs[i] consists of lowercase English letters.

https://leetcode.com/problems/group-anagrams/description/
'''

        def easy_solution(strs: List[str]) -> List[List[str]]:
            # Brute-force solution: Sort strings and group
            anagram_groups = {}
            for s in strs:
                sorted_s = ''.join(sorted(s))
                if sorted_s in anagram_groups:
                    anagram_groups[sorted_s].append(s)
                else:
                    anagram_groups[sorted_s] = [s]
            return list(anagram_groups.values())

        def optimized_solution(strs: List[str]) -> List[List[str]]:
            # Optimized solution: Use character count as key
            anagram_groups: Dict[str, List[str]] = {}
            for s in strs:
                count = [0] * 26
                for c in s:
                    count[ord(c) - ord('a')] += 1
                key = tuple(count)
                if key in anagram_groups:
                    anagram_groups[key].append(s)
                else:
                    anagram_groups[key] = [s]
            return list(anagram_groups.values())

        return Solution(
            question="Group Anagrams",
            easy_solution=easy_solution,
            optimized_solution=optimized_solution,
            time_complexity="O(n * k)",
            space_complexity="O(n * k)",
            similar_questions=["Valid Anagram", "Group Shifted Strings"],
            problem_statement=problem_statement
        )

    @staticmethod
    def valid_parentheses() -> Solution:
        """
        Generate a Solution object for the Valid Parentheses problem.

        Returns:
            Solution: A Solution object containing details of the problem.
        """
        def problem_statement() -> str:
            return '''Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

An input string is valid if:
    Open brackets must be closed by the same type of brackets.
    Open brackets must be closed in the correct order.
    Every close bracket has a corresponding open bracket of the same type.

Example 1:
Input: s = "()"
Output: true

Example 2:
Input: s = "()[]{}"
Output: true

Example 3:
Input: s = "(]"
Output: false

Constraints:
    1 <= s.length <= 104
    s consists of parentheses only '()[]{}'.

https://leetcode.com/problems/valid-parentheses/description/
'''
        def easy_solution(s: str) -> bool:
            # Brute-force solution: Use stack to check validity
            stack = []
            for char in s:
                if char in "({[":
                    stack.append(char)
                elif char in ")}]":
                    if not stack:
                        return False
                    if char == ")" and stack[-1] != "(":
                        return False
                    if char == "}" and stack[-1] != "{":
                        return False
                    if char == "]" and stack[-1] != "[":
                        return False
                    stack.pop()
            return len(stack) == 0

        def optimized_solution(s: str) -> bool:
            # Optimized solution: Use hashmap for mapping and stack
            stack = []
            mapping = {")": "(", "}": "{", "]": "["}
            for char in s:
                if char in mapping:
                    top_element = stack.pop() if stack else '#'
                    if mapping[char] != top_element:
                        return False
                else:
                    stack.append(char)
            return not stack

        return Solution(
            question="Valid Parentheses",
            easy_solution=easy_solution,
            optimized_solution=optimized_solution,
            time_complexity="O(n)",
            space_complexity="O(n)",
            similar_questions=["Generate Parentheses", "Longest Valid Parentheses", "Remove Invalid Parentheses"],
            problem_statement=problem_statement
        )

    @staticmethod
    def valid_palindrome() -> Solution:
        """
        Generate a Solution object for the Valid Palindrome problem.

        Returns:
            Solution: A Solution object containing details of the problem.
        """
        def problem_statement() -> str:
            return '''A phrase is a palindrome if, after converting all uppercase letters into lowercase letters and removing all non-alphanumeric characters, it reads the same forward and backward. Alphanumeric characters include letters and numbers.

Given a string s, return true if it is a palindrome, or false otherwise.

Example 1:
Input: s = "A man, a plan, a canal: Panama"
Output: true
Explanation: "amanaplanacanalpanama" is a palindrome.

Example 2:
Input: s = "race a car"
Output: false
Explanation: "raceacar" is not a palindrome.

Constraints:
    1 <= s.length <= 2 * 105
    s consists only of printable ASCII characters.

https://leetcode.com/problems/valid-palindrome/description/
'''

        def easy_solution(s: str) -> bool:
            # Brute-force solution: Clean string and check palindrome
            cleaned = ''.join(c.lower() for c in s if c.isalnum())
            return cleaned == cleaned[::-1]

        def optimized_solution(s: str) -> bool:
            # Optimized solution: Use two pointers
            left, right = 0, len(s) - 1
            while left < right:
                while left < right and not s[left].isalnum():
                    left += 1
                while left < right and not s[right].isalnum():
                    right -= 1
                if s[left].lower() != s[right].lower():
                    return False
                left += 1
                right -= 1
            return True

        return Solution(
            question="Valid Palindrome",
            easy_solution=easy_solution,
            optimized_solution=optimized_solution,
            time_complexity="O(n)",
            space_complexity="O(1)",
            similar_questions=["Palindrome Linked List", "Valid Palindrome II"],
            problem_statement=problem_statement
        )

    @staticmethod
    def longest_palindromic_substring() -> Solution:
        """
        Generate a Solution object for the Longest Palindromic Substring problem.

        Returns:
            Solution: A Solution object containing details of the problem.
        """
        def problem_statement() -> str:
            return '''Given a string s, return the longest palindromic substring in s.

Example 1:
Input: s = "babad"
Output: "bab"
Explanation: "aba" is also a valid answer.

Example 2:
Input: s = "cbbd"
Output: "bb"

Example 3:
Input: s = "a"
Output: "a"

Example 4:
Input: s = "ac"
Output: "a"

Constraints:
    1 <= s.length <= 1000
    s consist of only digits and English letters.

https://leetcode.com/problems/longest-palindromic-substring/description/
'''
        def easy_solution(s: str) -> str:
            # Brute-force solution: Check all substrings
            def is_palindrome(sub: str) -> bool:
                return sub == sub[::-1]

            longest = ""
            for i in range(len(s)):
                for j in range(i, len(s)):
                    substring = s[i:j+1]
                    if is_palindrome(substring) and len(substring) > len(longest):
                        longest = substring
            return longest

        def optimized_solution(s: str) -> str:
            # Optimized solution: Expand around center
            def expand_around_center(left: int, right: int) -> str:
                while left >= 0 and right < len(s) and s[left] == s[right]:
                    left -= 1
                    right += 1
                return s[left+1:right]

            if len(s) < 2:
                return s

            longest = ""
            for i in range(len(s)):
                palindrome1 = expand_around_center(i, i)
                palindrome2 = expand_around_center(i, i+1)
                longest = max(longest, palindrome1, palindrome2, key=len)
            return longest

        return Solution(
            question="Longest Palindromic Substring",
            easy_solution=easy_solution,
            optimized_solution=optimized_solution,
            time_complexity="O(n^2)",
            space_complexity="O(1)",
            similar_questions=["Shortest Palindrome", "Palindrome Permutation", "Palindrome Pairs"],
            problem_statement=problem_statement
        )

    @staticmethod
    def palindromic_substrings() -> Solution:
        """
        Generate a Solution object for the Palindromic Substrings problem.

        Returns:
            Solution: A Solution object containing details of the problem.
        """
        def problem_statement() -> str:
            return '''Given a string s, return the number of palindromic substrings in it.

A string is a palindrome when it reads the same backward as forward.

A substring is a contiguous sequence of characters within the string.

Example 1:
Input: s = "abc"
Output: 3
Explanation: Three palindromic strings: "a", "b", "c".

Example 2:
Input: s = "aaa"
Output: 6
Explanation: Six palindromic strings: "a", "a", "a", "aa", "aa", "aaa".

Constraints:
    1 <= s.length <= 1000
    s consists of lowercase English letters.

https://leetcode.com/problems/palindromic-substrings/description/
'''

        def easy_solution(s: str) -> int:
            # Brute-force solution: Check all substrings
            def is_palindrome(sub: str) -> bool:
                return sub == sub[::-1]

            count = 0
            for i in range(len(s)):
                for j in range(i, len(s)):
                    if is_palindrome(s[i:j+1]):
                        count += 1
            return count

        def optimized_solution(s: str) -> int:
            # Optimized solution: Expand around center
            def count_palindromes(left: int, right: int) -> int:
                count = 0
                while left >= 0 and right < len(s) and s[left] == s[right]:
                    count += 1
                    left -= 1
                    right += 1
                return count

            total_count = 0
            for i in range(len(s)):
                total_count += count_palindromes(i, i)  # Odd length palindromes
                total_count += count_palindromes(i, i+1)  # Even length palindromes
            return total_count

        return Solution(
            question="Palindromic Substrings",
            easy_solution=easy_solution,
            optimized_solution=optimized_solution,
            time_complexity="O(n^2)",
            space_complexity="O(1)",
            similar_questions=["Longest Palindromic Substring", "Longest Palindromic Subsequence"],
            problem_statement=problem_statement
        )

    @staticmethod
    def encode_and_decode_strings() -> Solution:
        """
        Generate a Solution object for the Encode and Decode Strings problem.

        Returns:
            Solution: A Solution object containing details of the problem.
        """
        def problem_statement() -> str:
            return '''Design an algorithm to encode a list of strings to a string. The encoded string is then sent over the network and is decoded back to the original list of strings.

Implement the encode and decode methods.

Example:
Input: ["Hello","World"]
Output: ["Hello","World"]

Note:
1. The string may contain any possible characters out of 256 valid ascii characters. Your algorithm should be generalized enough to work on any possible characters.
2. Do not use class member/global/static variables to store states. Your encode and decode algorithms should be stateless.

https://leetcode.com/problems/encode-and-decode-strings/description/
'''

        class Codec:
            def encode(self, strs: List[str]) -> str:
                # Encode by prefixing each string with its length and a special character
                return ''.join(f"{len(s)}#{s}" for s in strs)

            def decode(self, s: str) -> List[str]:
                # Decode by parsing the length and extracting the string
                result, i = [], 0
                while i < len(s):
                    j = s.index('#', i)
                    length = int(s[i:j])
                    result.append(s[j+1:j+1+length])
                    i = j + 1 + length
                return result

        def easy_solution(strs: List[str]) -> List[str]:
            # Utilize the Codec class for encoding and decoding
            codec = Codec()
            encoded = codec.encode(strs)
            return codec.decode(encoded)

        def optimized_solution(strs: List[str]) -> List[str]:
            # The easy solution is already optimal
            return easy_solution(strs)

        return Solution(
            question="Encode and Decode Strings",
            easy_solution=easy_solution,
            optimized_solution=optimized_solution,
            time_complexity="O(n)",
            space_complexity="O(n)",
            similar_questions=["Count and Say", "Serialize and Deserialize Binary Tree"],
            problem_statement=problem_statement
        )

