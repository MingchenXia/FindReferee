from __future__ import annotations

import unittest
from pathlib import Path


HTML = (Path(__file__).resolve().parents[1] / "static" / "index.html").read_text(
    encoding="utf-8"
)


class FrontendContractTests(unittest.TestCase):
    def test_minimal_sol_xhigh_are_defaults(self) -> None:
        self.assertIn('<body class="minimal-mode">', HTML)
        self.assertIn('class="active" data-view="minimal" aria-pressed="true"', HTML)
        self.assertIn('<option value="gpt-5.6-sol" selected>', HTML)
        self.assertIn('<option value="xhigh" selected>', HTML)

    def test_completed_result_exposes_explicit_consent_share_control(self) -> None:
        initial_result = HTML.split('<section class="card result-pane" id="result">', 1)[1].split(
            "</section>", 1
        )[0]
        self.assertNotIn("share-maintainer", initial_result)
        self.assertIn('data-action="share-maintainer"', HTML)
        self.assertIn("window.confirm", HTML)
        self.assertIn("X-Unsent: 1", HTML)
        self.assertIn("xiamingchen2008@gmail.com", HTML)

    def test_share_copy_states_the_restricted_use(self) -> None:
        self.assertGreaterEqual(
            HTML.count(
                "used only to improve FindReferee and will not be distributed to any third party"
            ),
            2,
        )
        self.assertIn("The website has not uploaded or sent anything", HTML)

    def test_manuscript_author_screen_is_visible_and_namesake_safe(self) -> None:
        self.assertIn('id="underlying-authors"', HTML)
        self.assertIn("Add one manuscript author per line", HTML)
        self.assertIn("official profile URL in Context", HTML)
        self.assertNotIn("Verified manuscript authors are excluded from referee scoring", HTML)
        self.assertNotIn("A shared name alone never excludes a candidate", HTML)
        self.assertIn("Manuscript-author eligibility", HTML)

    def test_attribution_copy_uses_mathematician_examples_without_unknown_label(self) -> None:
        self.assertIn("Who wrote this report?", HTML)
        self.assertNotIn("Who might have written this report?", HTML)
        self.assertIn(
            'placeholder="Carl Friedrich Gauss&#10;Leonhard Euler / Leonhard Paul Euler',
            HTML,
        )
        self.assertNotIn("Carl Friedrich Gauss /", HTML)
        self.assertIn("Leonhard Euler / Leonhard Paul Euler", HTML)
        self.assertIn("Sofia Kovalevskaya / Sofya Kovalevskaya", HTML)
        self.assertIn(
            'placeholder="Bernhard Riemann&#10;Évariste Galois / Evariste Galois',
            HTML,
        )
        self.assertNotIn("Bernhard Riemann /", HTML)
        self.assertIn("Évariste Galois / Evariste Galois", HTML)
        self.assertNotIn("Unknown author", HTML)

    def test_primary_action_uses_one_plain_label_in_every_state(self) -> None:
        self.assertIn('id="submit" type="submit">Let\'s go!</button>', HTML)
        self.assertNotIn("Analyze automatically with Codex", HTML)
        self.assertNotIn("Compare automatically with Codex", HTML)
        self.assertNotIn("Discover authors automatically with Codex", HTML)
        self.assertNotIn("Analyzing…", HTML)

    def test_both_document_uploads_share_the_same_drop_zone_format(self) -> None:
        self.assertIn(".drop input[type=file] { display:none; }", HTML)
        self.assertNotIn("#file-input { display:none; }", HTML)
        self.assertIn('class="drop" id="drop"><input id="file-input"', HTML)
        self.assertIn(
            'class="drop" id="subject-drop"><input id="subject-file-input"',
            HTML,
        )
        self.assertNotIn("<label>Or upload document(s)</label>", HTML)
        self.assertLess(
            HTML.index("<label>Upload document(s)</label>"),
            HTML.index('<label for="text-input">Or paste text'),
        )

    def test_candidate_discovery_hint_starts_on_a_new_line(self) -> None:
        self.assertIn(
            "same author.<br>Leave blank to let AI discover likely public authors first.",
            HTML,
        )

    def test_reviewed_work_uses_plain_paper_or_book_label(self) -> None:
        self.assertIn("Paper/Book under review", HTML)
        self.assertIn("(optional, but recommended)", HTML)
        self.assertNotIn("(optional for referee reports)", HTML)
        self.assertNotIn("Underlying document", HTML)
        self.assertNotIn("underlying document", HTML)

    def test_candidate_scope_switch_is_on_by_default_and_sent_to_backend(self) -> None:
        self.assertIn(
            'id="explore-outside-candidates" type="checkbox"', HTML
        )
        self.assertIn('aria-describedby="outside-search-help" checked', HTML)
        self.assertIn("Explore beyond my candidate list", HTML)
        self.assertIn("Candidate list is empty, so AI discovery is required", HTML)
        self.assertIn("data.append('explore_outside_candidates'", HTML)
        self.assertIn("Candidate search scope", HTML)
        self.assertIn("supplied shortlist only; the outside probability is unnamed", HTML)
        self.assertIn("method-specific expertise", HTML)
        self.assertIn("normalized.includes('adjudication')", HTML)


if __name__ == "__main__":
    unittest.main()
