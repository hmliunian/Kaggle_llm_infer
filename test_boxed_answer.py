import unittest

import official_metric
import train_sft


# Answers (including ones with literal braces / backslashes and unbalanced braces)
# that exercise the raw, official-metric-aligned boxed handling.
ANSWERS = [
    "plain",
    "42",
    "{left",
    "right}",
    "a{b}c",
    "+}",
    "{17",
    "-}",
    r"\![<_",
    r"\path{x}",
    r"\\",
    "}52",
]


class BoxedAnswerTest(unittest.TestCase):
    def _target(self, answer):
        # Exactly how training writes the supervised final-answer span (raw answer).
        return train_sft.format_answer_text(answer)

    def test_raw_target_round_trips_through_official_metric(self):
        # The whole point of removing escaping: the official grader must read back
        # the exact gold answer from a raw \boxed{answer}. target.
        for answer in ANSWERS:
            with self.subTest(answer=answer):
                target = self._target(answer)
                self.assertEqual(
                    official_metric.extract_final_answer(target).strip(),
                    str(answer).strip(),
                )

    def test_local_extract_matches_official_extract(self):
        for answer in ANSWERS:
            with self.subTest(answer=answer):
                target = self._target(answer)
                self.assertEqual(
                    train_sft.extract_boxed(target),
                    official_metric.extract_final_answer(target).strip(),
                )

    def test_truncate_keeps_full_boxed_answer(self):
        # Trailing reasoning/ramble after the "}." terminator must be dropped, but
        # the boxed answer (incl. unbalanced braces) must survive for the grader.
        for answer in ANSWERS:
            with self.subTest(answer=answer):
                decoded = self._target(answer) + " then the model keeps rambling }"
                truncated = train_sft.truncate_after_first_boxed(decoded)
                self.assertEqual(
                    official_metric.extract_final_answer(truncated).strip(),
                    str(answer).strip(),
                )

    def test_extract_picks_last_boxed_when_reasoning_has_boxed_text(self):
        text = r"draft \boxed{wrong} final \boxed{x{y}}."
        self.assertEqual(train_sft.extract_boxed(text), "x{y}")
        self.assertEqual(
            official_metric.extract_final_answer(text).strip(), "x{y}"
        )

    def test_extract_boxed_returns_none_without_box(self):
        self.assertIsNone(train_sft.extract_boxed("The final answer is 7"))


if __name__ == "__main__":
    unittest.main()
