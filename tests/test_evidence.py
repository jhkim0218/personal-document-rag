from __future__ import annotations

import unittest
from unittest.mock import patch

from rag.answer import Answerer, MAX_GENERATED_ANSWER_CHARACTERS, validate_citations, detect_possible_conflict
from dataclasses import replace
from tests.test_answer import result


class EvidenceTests(unittest.TestCase):
    def test_late_support_is_selected_and_offsets_identify_literal_quote(self):
        text = "Background unrelated material. " * 21 + "The launch date is 2026-10-01."
        answer = Answerer(api_key="").answer("What is the launch date?", [result(text)])
        source = answer.sources[0]
        self.assertGreater(source["quote_start"], 500)
        self.assertIn("2026-10-01", answer.text)
        self.assertEqual(text[source["quote_start"]:source["quote_end"]], source["excerpt"])
        self.assertEqual(source["context"], text)

    def test_api_context_is_not_the_display_excerpt_and_timeout_preserves_quotes(self):
        text = "Background material. " * 30 + "The launch is Friday."
        with patch.object(Answerer, "_openai_answer", side_effect=TimeoutError("test timeout")) as generate:
            answer = Answerer(api_key="mock-only").answer("When is launch?", [result(text)])
        self.assertEqual(generate.call_args.args[1][0]["context"], text)
        self.assertIn("Friday", answer.text)
        self.assertEqual(answer.error, "test timeout")
        self.assertIsNone(validate_citations(answer.text.split('\n',1)[1], len(answer.sources)))

    def test_invalid_model_citations_fall_back_to_same_source_list(self):
        with patch.object(Answerer, "_openai_answer", return_value="Invented [99]"):
            answer = Answerer(api_key="mock-only").answer("When is launch?", [result("A preamble. The launch is Friday.")])
        self.assertEqual(answer.status, "fallback")
        self.assertNotIn("99", answer.text)
        self.assertIn("The launch is Friday. [1]", answer.text)
        self.assertEqual(len(answer.sources), 1)

    def test_uncited_second_claim_rejected_even_with_valid_first_citation(self):
        self.assertIsNotNone(validate_citations("The launch is Friday. [1] The budget is 100.", 1))
        self.assertIsNone(validate_citations("The launch is Friday. [1] The owner is Mina. [2]", 2))

    def test_metadata_match_alone_does_not_create_body_evidence(self):
        answer = Answerer(api_key="").answer("Annual revenue?", [result("The launch is Friday.", overlap=1.0, keyword_score=10.0)])
        self.assertEqual(answer.status, "abstained")

    def test_related_launch_document_without_requested_date_abstains(self):
        answer = Answerer(api_key="").answer("What is the launch date?", [result("The launch requires an approval. The launch date is not recorded.")])
        self.assertEqual(answer.status, "abstained")

    def test_different_metrics_and_different_projects_are_not_conflicts(self):
        first = result("Orion p50 latency is 100ms.")
        second = replace(first, path="other.md", chunk_id="other", text="Orion p95 latency is 200ms.")
        self.assertIsNone(detect_possible_conflict("What are the latency metrics?", [first, second]))
        first = replace(first, text="Orion launch date is 2026-10-01.")
        second = replace(second, text="Vega launch date is 2026-10-08.")
        self.assertIsNone(detect_possible_conflict("What is the launch date?", [first, second]))
        second = replace(second, text="Orion launch date is 2026-10-08.")
        self.assertIsNotNone(detect_possible_conflict("What is the launch date?", [first, second]))

    def test_explicit_multi_fact_request_is_partial_when_a_required_fact_is_missing(self):
        incomplete = Answerer(api_key="").answer(
            "What changed before and after, and why?",
            [result("The deployment changed from 2026-04-10 to 2026-04-17.")],
        )
        self.assertEqual(incomplete.status, "partial")
        self.assertIn("변경 이유", incomplete.text)

        complete = Answerer(api_key="").answer(
            "What changed before and after, and why?",
            [result("The deployment changed from 2026-04-10 to 2026-04-17 because approval was delayed.")],
        )
        self.assertEqual(complete.status, "changed")

    def test_requested_p50_and_p95_do_not_pass_with_one_metric(self):
        answer = Answerer(api_key="").answer("What are p50 and p95 latency?", [result("The p50 latency is 100ms.")])
        self.assertEqual(answer.status, "partial")
        self.assertIn("p95", answer.text)

    def test_single_explicit_change_record_is_structured_without_claiming_latest(self):
        answer = Answerer(api_key="").answer(
            "What changed before and after, and why?",
            [result("2026-04-02: rollout 2026-04-10 → 2026-04-17 changed because approval was delayed.")],
        )
        self.assertEqual(answer.status, "changed")
        self.assertIn("변경 전: rollout 2026-04-10", answer.text)
        self.assertIn("- 2026-04-02: rollout 2026-04-10 → 2026-04-17 changed because approval was delayed. [1]", answer.text)
        self.assertIn("최신 결정이라고 단정하지", answer.text)
        self.assertEqual(answer.sources[0]["excerpt"], "2026-04-02: rollout 2026-04-10 → 2026-04-17 changed because approval was delayed.")

    def test_change_answer_ignores_related_change_mentions_without_before_after_values(self):
        answer = Answerer(api_key="").answer(
            "오로라 배포일 변경 전후와 변경 이유는?",
            [result("변경 검토를 시작했다. 배포일을 9월 10일에서 9월 17일로 변경한다. 변경 이유는 승인 지연이다.")],
        )
        self.assertEqual(answer.status, "changed")
        self.assertIn("이유: 승인 지연이다", answer.text)
        self.assertIn("- 변경 이유는 승인 지연이다. [1]", answer.text)

    def test_oversized_generated_answer_falls_back_to_the_same_evidence(self):
        answerer = Answerer(api_key="mock-only")
        body = {"output": [{"type": "message", "content": [{"type": "output_text", "text": "A" * (MAX_GENERATED_ANSWER_CHARACTERS + 1)}]}]}
        with patch.object(answerer.requests, "send", return_value=body):
            answer = answerer.answer("When is launch?", [result("The launch is Friday.")])
        self.assertEqual(answer.status, "fallback")
        self.assertIn("Friday", answer.text)
        self.assertIn("character limit", answer.error)


if __name__ == "__main__":
    unittest.main()
