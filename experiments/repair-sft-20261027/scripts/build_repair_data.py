"""Build a small authored repair corpus. No task code is executed on the host."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYSTEM = "Fix the Python function. Return only complete Python code, without Markdown fences."


def task(name, split, requirement, signature, body, bugs, cases):
    return dict(source_task_id="authored-v1-" + name, family=name, split=split,
                requirement=requirement, signature=signature,
                reference=signature + "\n" + body,
                bugs=[dict(kind=k, code=signature + "\n" + b) for k, b in bugs],
                cases=[dict(args=a, expected=e, visibility="public" if i == 0 else "hidden")
                       for i, (a, e) in enumerate(cases)])


def catalog():
    # Expected outputs are independently specified literals, not reference execution.
    return [
        task("stable_unique", "train", "Return distinct integers in their first-occurrence order. Do not modify items.",
             "def stable_unique(items):", "    return list(dict.fromkeys(items))",
             [("sorting_loses_order", "    return sorted(set(items))"),
              ("duplicates_kept", "    return list(items)"),
              ("last_item_lost", "    return list(dict.fromkeys(items[:-1]))")],
             [([[3, 1, 3, 2]], [3, 1, 2]), ([[]], []), ([[4]], [4]),
              ([[2, 2, 2]], [2]), ([[-1, 0, -1, 2]], [-1, 0, 2]),
              ([[5, 4, 3]], [5, 4, 3])]),
        task("inclusive_sum", "train", "Return the sum of integers from lo through hi inclusive. Return 0 if lo > hi.",
             "def inclusive_sum(lo, hi):", "    return sum(range(lo, hi + 1))",
             [("upper_excluded", "    return sum(range(lo, hi))"),
              ("lower_excluded", "    return sum(range(lo + 1, hi + 1))"),
              ("reversed_bounds", "    return sum(range(hi, lo + 1))")],
             [([2, 5], 14), ([0, 0], 0), ([4, 4], 4), ([5, 2], 0),
              ([-3, -1], -6), ([-2, 2], 0), ([1, 10], 55)]),
        task("chunk_list", "train", "Split items into consecutive lists of size k, retaining a shorter final chunk. k is a positive integer.",
             "def chunk_list(items, k):", "    return [items[i:i + k] for i in range(0, len(items), k)]",
             [("tail_dropped", "    return [items[i:i + k] for i in range(0, len(items) - k + 1, k)]"),
              ("stride_one", "    return [items[i:i + k] for i in range(len(items))]"),
              ("slice_short", "    return [items[i:i + k - 1] for i in range(0, len(items), k)]")],
             [([[1, 2, 3, 4, 5], 2], [[1, 2], [3, 4], [5]]), ([[], 3], []),
              ([[1], 5], [[1]]), ([[1, 2], 1], [[1], [2]]),
              ([[0, 0, 0, 0], 2], [[0, 0], [0, 0]]),
              ([[1, 2, 3], 3], [[1, 2, 3]])]),
        task("count_vowels", "train", "Count ASCII vowels a,e,i,o,u in text, ignoring case. Other characters are not vowels.",
             "def count_vowels(text):", "    return sum(c in 'aeiou' for c in text.lower())",
             [("case_sensitive", "    return sum(c in 'aeiou' for c in text)"),
              ("u_missing", "    return sum(c in 'aeio' for c in text.lower())"),
              ("distinct_only", "    return len(set(text.lower()) & set('aeiou'))")],
             [(["AEiouu"], 6), ([""], 0), (["rhythm"], 0), (["banana"], 3),
              (["U"], 1), (["é123A!"], 1), (["aAa"], 3)]),
        task("balanced_parentheses", "train", "Return whether parentheses in text are balanced and correctly ordered. Ignore all non-parenthesis characters.",
             "def balanced_parentheses(text):",
             "    depth = 0\n    for c in text:\n        depth += (c == '(') - (c == ')')\n        if depth < 0:\n            return False\n    return depth == 0",
             [("only_counts", "    return text.count('(') == text.count(')')"),
              ("prefix_not_checked", "    depth = 0\n    for c in text:\n        depth += (c == '(') - (c == ')')\n    return depth == 0"),
              ("unclosed_and_misordered_accepted", "    depth = 0\n    for c in text:\n        depth += (c == '(') - (c == ')')\n    return depth >= 0")],
             [(["a)(b"], False), ([""], True), (["abc"], True), (["(a(b)c)"], True),
              (["(()"], False), ([")("], False), (["()()"], True), (["("], False)]),
        task("run_length", "train", "Encode consecutive equal characters as [character, count] pairs, preserving run order.",
             "def run_length(text):",
             "    result = []\n    for c in text:\n        if result and result[-1][0] == c:\n            result[-1][1] += 1\n        else:\n            result.append([c, 1])\n    return result",
             [("global_counts", "    return [[c, text.count(c)] for c in dict.fromkeys(text)]"),
              ("counts_start_zero", "    result = []\n    for c in text:\n        if result and result[-1][0] == c:\n            result[-1][1] += 1\n        else:\n            result.append([c, 0])\n    return result"),
              ("last_character_lost", "    result = []\n    for c in text[:-1]:\n        if result and result[-1][0] == c:\n            result[-1][1] += 1\n        else:\n            result.append([c, 1])\n    return result")],
             [(["aaba"], [["a", 2], ["b", 1], ["a", 1]]), ([""], []),
              (["x"], [["x", 1]]), (["aaaa"], [["a", 4]]),
              (["ab"], [["a", 1], ["b", 1]]), (["  !"], [[" ", 2], ["!", 1]])]),
        task("flatten_once", "train", "Flatten a list of lists by exactly one level. Preserve order, duplicates and all elements.",
             "def flatten_once(groups):", "    return [item for group in groups for item in group]",
             [("first_group_only", "    return list(groups[0]) if groups else []"),
              ("empty_stops_loop", "    result = []\n    for group in groups:\n        if not group:\n            break\n        result.extend(group)\n    return result"),
              ("drops_falsy", "    return [item for group in groups for item in group if item]")],
             [([[[], [0, 1], [2]]], [0, 1, 2]), ([[]], []), ([[[]]], []),
              ([[[1, 1], [2]]], [1, 1, 2]), ([[[[1]], [None, False]]], [[1], None, False]),
              ([[[1], [], [2]]], [1, 2])]),
        task("transpose", "train", "Transpose a rectangular matrix (list of equal-length rows). Return lists of columns. Empty input or zero columns yields [].",
             "def transpose(matrix):", "    return [list(col) for col in zip(*matrix)]",
             [("identity", "    return [list(row) for row in matrix]"),
              ("drops_last_column", "    return [list(col) for col in zip(*matrix)][:-1]"),
              ("reverse_columns", "    return [list(col) for col in zip(*matrix)][::-1]")],
             [([[[1, 2, 3], [4, 5, 6]]], [[1, 4], [2, 5], [3, 6]]),
              ([[]], []), ([[[], []]], []), ([[[1]]], [[1]]),
              ([[[1], [2], [3]]], [[1, 2, 3]]), ([[[0, -1]]], [[0], [-1]])]),
        task("merge_intervals", "train", "Merge closed intervals [start,end], where start <= end. Touching intervals merge. Return intervals sorted by start; do not modify input.",
             "def merge_intervals(intervals):",
             "    merged = []\n    for start, end in sorted(intervals):\n        if merged and start <= merged[-1][1]:\n            merged[-1][1] = max(merged[-1][1], end)\n        else:\n            merged.append([start, end])\n    return merged",
             [("touch_not_merged", "    merged = []\n    for start, end in sorted(intervals):\n        if merged and start < merged[-1][1]:\n            merged[-1][1] = max(merged[-1][1], end)\n        else:\n            merged.append([start, end])\n    return merged"),
              ("nested_shrinks_end", "    merged = []\n    for start, end in sorted(intervals):\n        if merged and start <= merged[-1][1]:\n            merged[-1][1] = end\n        else:\n            merged.append([start, end])\n    return merged"),
              ("unsorted_input", "    merged = []\n    for start, end in intervals:\n        if merged and start <= merged[-1][1]:\n            merged[-1][1] = max(merged[-1][1], end)\n        else:\n            merged.append([start, end])\n    return merged")],
             [([[[5, 7], [1, 5], [2, 3]]], [[1, 7]]), ([[]], []),
              ([[[1, 1]]], [[1, 1]]), ([[[1, 10], [2, 3]]], [[1, 10]]),
              ([[[1, 2], [4, 5]]], [[1, 2], [4, 5]]),
              ([[[-3, 0], [0, 2], [2, 4]]], [[-3, 4]])]),
        task("binary_search_first", "train", "In sorted integer items, return the first index of target, or -1 if absent.",
             "def binary_search_first(items, target):",
             "    lo, hi = 0, len(items)\n    while lo < hi:\n        mid = (lo + hi) // 2\n        if items[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid\n    return lo if lo < len(items) and items[lo] == target else -1",
             [("last_match", "    return len(items) - 1 - items[::-1].index(target) if target in items else -1"),
              ("insertion_not_absent", "    import bisect\n    return bisect.bisect_left(items, target)"),
              ("one_based", "    return items.index(target) + 1 if target in items else -1")],
             [([[1, 2, 2, 4], 2], 1), ([[], 0], -1), ([[1, 2], 3], -1),
              ([[1, 2], 0], -1), ([[3], 3], 0), ([[-2, -2, 0], -2], 0),
              ([[1, 3, 5], 4], -1)]),
        task("rotate_right", "dev", "Rotate items right by k positions. k can be any integer; negative k rotates left. Empty items gives [].",
             "def rotate_right(items, k):",
             "    if not items:\n        return []\n    k %= len(items)\n    return items[-k:] + items[:-k]",
             [("direction_reversed", "    if not items:\n        return []\n    k %= len(items)\n    return items[k:] + items[:k]")],
             [([[1, 2, 3], 1], [3, 1, 2]), ([[], 4], []), ([[1], -8], [1]),
              ([[1, 2, 3], 0], [1, 2, 3]), ([[1, 2, 3], 4], [3, 1, 2]),
              ([[1, 2, 3], -1], [2, 3, 1])]),
        task("anagram", "dev", "Return whether two strings are anagrams, ignoring whitespace and case. Punctuation remains significant.",
             "def anagram(left, right):",
             "    def normalized(s):\n        return sorted(c for c in s.casefold() if not c.isspace())\n    return normalized(left) == normalized(right)",
             [("ignores_multiplicity", "    return {c for c in left.casefold() if not c.isspace()} == {c for c in right.casefold() if not c.isspace()}")],
             [(["aab", "abb"], False), (["a ab", "ab a"], True), (["", " \t"], True),
              (["Dormitory", "Dirty room"], True), (["a!", "a"], False),
              (["ABC", "cba"], True), (["ß", "ss"], True)]),
        task("gcd", "dev", "Return the nonnegative greatest common divisor of integer a,b. gcd(0,0) is 0.",
             "def gcd(a, b):",
             "    a, b = abs(a), abs(b)\n    while b:\n        a, b = b, a % b\n    return a",
             [("sign_not_normalized", "    while b:\n        a, b = b, a % b\n    return a")],
             [([-12, -8], 4), ([0, 0], 0), ([0, -7], 7), ([13, 5], 1),
              ([18, 12], 6), ([-9, 0], 9), ([4, 4], 4)]),
        task("invert_mapping", "dev", "Invert a dictionary with string keys and string values: map each value to a list of its keys in input insertion order.",
             "def invert_mapping(mapping):",
             "    result = {}\n    for key, value in mapping.items():\n        result.setdefault(value, []).append(key)\n    return result",
             [("duplicate_values_overwrite", "    return {value: [key] for key, value in mapping.items()}")],
             [([{"a": "x", "b": "x", "c": "y"}], {"x": ["a", "b"], "y": ["c"]}),
              ([{}], {}), ([{"a": "x"}], {"x": ["a"]}),
              ([{"z": "", "a": ""}], {"": ["z", "a"]}),
              ([{"a": "b", "b": "a"}], {"b": ["a"], "a": ["b"]}),
              ([{"a": "x", "b": "y", "c": "x"}], {"x": ["a", "c"], "y": ["b"]})]),
        task("matrix_diagonal", "dev", "Return the sum of the main and anti-diagonals of a square integer matrix, counting the center once. Empty matrix gives 0.",
             "def matrix_diagonal(matrix):",
             "    n = len(matrix)\n    total = sum(matrix[i][i] + matrix[i][n - 1 - i] for i in range(n))\n    if n % 2:\n        total -= matrix[n // 2][n // 2]\n    return total",
             [("center_twice", "    n = len(matrix)\n    return sum(matrix[i][i] + matrix[i][n - 1 - i] for i in range(n))")],
             [([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], 25), ([[]], 0),
              ([[[7]]], 7), ([[[1, 2], [3, 4]]], 10),
              ([[[0, 0, 0], [0, -2, 0], [0, 0, 0]]], -2),
              ([[[1, 1, 1, 1]] * 4], 8)]),
        task("longest_common_prefix", "test", "Return the longest prefix common to all strings in words. Empty words gives ''. Case is significant.",
             "def longest_common_prefix(words):",
             "    if not words:\n        return ''\n    prefix = words[0]\n    for word in words[1:]:\n        while not word.startswith(prefix):\n            prefix = prefix[:-1]\n    return prefix",
             [("first_word_only", "    return words[0] if words else ''")],
             [([["flower", "flow", "flight"]], "fl"), ([[]], ""), ([["abc"]], "abc"),
              ([["", "abc"]], ""), ([["dog", "cat"]], ""), ([["ab", "abc", "ab"]], "ab"),
              ([["A", "a"]], "")]),
        task("frequency_winner", "test", "Return the most frequent integer in items; break ties by smallest integer. Empty items gives None.",
             "def frequency_winner(items):",
             "    if not items:\n        return None\n    from collections import Counter\n    counts = Counter(items)\n    return min(counts, key=lambda x: (-counts[x], x))",
             [("first_tie_wins", "    if not items:\n        return None\n    from collections import Counter\n    return Counter(items).most_common(1)[0][0]")],
             [([[3, 1, 3, 1]], 1), ([[]], None), ([[9]], 9),
              ([[2, 2, 1]], 2), ([[-1, -2]], -2), ([[0, 1, 0]], 0)]),
        task("max_subarray", "test", "Return the largest sum of a nonempty contiguous subarray. Empty items gives 0.",
             "def max_subarray(items):",
             "    if not items:\n        return 0\n    best = current = items[0]\n    for x in items[1:]:\n        current = max(x, current + x)\n        best = max(best, current)\n    return best",
             [("empty_subarray_allowed", "    best = current = 0\n    for x in items:\n        current = max(0, current + x)\n        best = max(best, current)\n    return best")],
             [([[-5, -2, -7]], -2), ([[]], 0), ([[4]], 4),
              ([[-2, 1, -3, 4, -1, 2, 1, -5, 4]], 6),
              ([[1, 2, 3]], 6), ([[0, -1, 0]], 0)]),
        task("is_prime", "test", "Return whether integer n is prime. Integers below 2 are not prime.",
             "def is_prime(n):",
             "    if n < 2:\n        return False\n    d = 2\n    while d * d <= n:\n        if n % d == 0:\n            return False\n        d += 1\n    return True",
             [("square_boundary_excluded", "    if n < 2:\n        return False\n    d = 2\n    while d * d < n:\n        if n % d == 0:\n            return False\n        d += 1\n    return True")],
             [([49], False), ([-1], False), ([0], False), ([1], False), ([2], True),
              ([3], True), ([4], False), ([97], True), ([121], False), ([15], False)]),
        task("edit_distance", "test", "Return Levenshtein distance between strings a,b; insertion, deletion and substitution each cost 1.",
             "def edit_distance(a, b):",
             "    row = list(range(len(b) + 1))\n    for i, x in enumerate(a, 1):\n        new = [i]\n        for j, y in enumerate(b, 1):\n            new.append(min(new[-1] + 1, row[j] + 1, row[j - 1] + (x != y)))\n        row = new\n    return row[-1]",
             [("substitution_cost_two", "    row = list(range(len(b) + 1))\n    for i, x in enumerate(a, 1):\n        new = [i]\n        for j, y in enumerate(b, 1):\n            new.append(min(new[-1] + 1, row[j] + 1, row[j - 1] + 2 * (x != y)))\n        row = new\n    return row[-1]")],
             [(["cat", "cut"], 1), (["", ""], 0), (["abc", ""], 3),
              (["", "ab"], 2), (["kitten", "sitting"], 3), (["same", "same"], 0),
              (["ab", "ba"], 2)]),
    ]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def main():
    tasks = catalog()
    # This bug needs an absent-target witness, while the other two need duplicates.
    search = next(t for t in tasks if t["family"] == "binary_search_first")
    search["bugs"][1]["public_case_index"] = 2
    search["cases"][2]["visibility"] = "public"
    # Partition is determined by the authored catalog before variants are expanded.
    write_json(ROOT / "configs/repair_source_catalog.json", tasks)
    write_json(ROOT / "data/repair/source_partition.json",
               {t["source_task_id"]: t["split"] for t in tasks})
    for t in tasks:
        ast.parse(t["reference"])
        for b in t["bugs"]:
            ast.parse(b["code"])
    print(f"Catalog frozen: {len(tasks)} tasks. Run verify_repair_data.py in Docker before JSONL publication.")


if __name__ == "__main__":
    main()
