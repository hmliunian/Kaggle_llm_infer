import unittest

import train_sft


class BoxedAnswerTest(unittest.TestCase):
    def test_escape_round_trips_braces_and_backslashes(self):
        answers = [
            "plain",
            "{left",
            "right}",
            "a{b}c",
            r"\path\{x\}",
            r"\\",
            r"}\{",
        ]

        for answer in answers:
            with self.subTest(answer=answer):
                boxed = rf"\boxed{{{train_sft.escape_boxed_answer(answer)}}}."
                self.assertEqual(train_sft.extract_boxed(boxed), answer)

    def test_stop_parser_ignores_answer_internal_escaped_braces(self):
        text = r"</think>\nThe final answer is \boxed{a\}b\{c} trailing"

        self.assertEqual(train_sft.extract_boxed(text), "a}b{c")
        self.assertEqual(
            train_sft.truncate_after_first_boxed(text),
            r"</think>\nThe final answer is \boxed{a\}b\{c}",
        )

    def test_extracts_last_boxed_answer_when_reasoning_contains_boxed_text(self):
        text = r"draft \boxed{wrong} final \boxed{x\{y\}}."

        self.assertEqual(train_sft.extract_boxed(text), "x{y}")

    def test_unclosed_box_is_not_parsed(self):
        self.assertIsNone(train_sft.extract_boxed(r"The final answer is \boxed{a\}b"))


if __name__ == "__main__":
    unittest.main()
