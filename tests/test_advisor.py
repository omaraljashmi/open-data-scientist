from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import unittest

import pandas as pd
import requests

from ods import AdvisorError, dataset_brief, request_chart_advice


SECRET = "SECRET-CELL-VALUE-9481"


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "segment": ["Enterprise", SECRET, "Consumer", "Consumer", "Enterprise"],
            "monthly_spend": [4200.0, 850.0, 120.0, 300.0, 5100.0],
            "signup_date": pd.to_datetime(
                ["2026-01-02", "2026-02-11", "2026-03-09", "2026-04-28", "2026-05-15"]
            ),
        }
    )


class _StubHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length)
        self.server.seen.append(  # type: ignore[attr-defined]
            {
                "path": self.path,
                "headers": {key: value for key, value in self.headers.items()},
                "body": json.loads(body),
            }
        )
        responses = self.server.responses  # type: ignore[attr-defined]
        status, payload = responses.pop(0) if responses else (200, _chat_response("{}"))
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, *args: object) -> None:
        return


def _chat_response(content: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


class AdvisorStubTests(unittest.TestCase):
    def setUp(self) -> None:
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _StubHandler)
        self.server.seen = []  # type: ignore[attr-defined]
        self.server.responses = []  # type: ignore[attr-defined]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        port = self.server.server_address[1]
        self.base_url = f"http://localhost:{port}".replace("localhost", "localhost")
        self.base_url = f"http://localhost:{port}"
        self.session = requests.Session()
        self.session.trust_env = False

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def _ask(self, frame: pd.DataFrame):
        return request_chart_advice(
            frame,
            intent="Overview",
            base_url=self.base_url,
            model="stub-model",
            api_key="free-tier-key",
            session=self.session,
        )

    def test_request_carries_metadata_only_never_cell_values(self) -> None:
        content = json.dumps(
            {
                "charts": [
                    {
                        "kind": "category_aggregate",
                        "title": "Spend by segment",
                        "explanation": "Compares average spend across segments.",
                        "x": "segment",
                        "y": "monthly_spend",
                        "aggregation": "mean",
                    }
                ]
            }
        )
        self.server.responses = [(200, _chat_response(content))]  # type: ignore[attr-defined]
        suggestions = self._ask(_sample_frame())

        sent = json.dumps(self.server.seen[0]["body"])  # type: ignore[attr-defined]
        self.assertNotIn(SECRET, sent)
        self.assertNotIn("4200", sent)  # numeric cell values stay local too
        self.assertIn("segment", sent)  # column names are the metadata we do send
        self.assertEqual(
            self.server.seen[0]["headers"]["Authorization"],  # type: ignore[attr-defined]
            "Bearer free-tier-key",
        )
        self.assertEqual(len(suggestions), 1)
        self.assertEqual(suggestions[0].kind, "category_aggregate")
        self.assertEqual(suggestions[0].x, "segment")
        self.assertEqual(suggestions[0].y, "monthly_spend")

    def test_invalid_columns_and_kinds_are_dropped(self) -> None:
        content = json.dumps(
            {
                "charts": [
                    {"kind": "scatter", "x": "segment", "y": "monthly_spend"},  # x not numeric
                    {"kind": "histogram", "x": "no_such_column"},
                    {"kind": "pie", "x": "segment"},  # unsupported kind
                    {
                        "kind": "time_series",
                        "title": "Spend over time",
                        "explanation": "Average spend per month.",
                        "x": "signup_date",
                        "y": "monthly_spend",
                        "aggregation": "mean",
                        "date_grain": "month",
                    },
                ]
            }
        )
        self.server.responses = [(200, _chat_response(content))]  # type: ignore[attr-defined]
        suggestions = self._ask(_sample_frame())
        self.assertEqual(len(suggestions), 1)
        self.assertEqual(suggestions[0].kind, "time_series")
        self.assertEqual(suggestions[0].date_grain, "month")

    def test_fenced_json_is_tolerated(self) -> None:
        fenced = "```json\n" + json.dumps(
            {"charts": [{"kind": "category_count", "x": "segment"}]}
        ) + "\n```"
        self.server.responses = [(200, _chat_response(fenced))]  # type: ignore[attr-defined]
        suggestions = self._ask(_sample_frame())
        self.assertEqual(suggestions[0].kind, "category_count")

    def test_auth_failure_has_actionable_message(self) -> None:
        self.server.responses = [(401, {"error": "bad key"})]  # type: ignore[attr-defined]
        with self.assertRaises(AdvisorError) as caught:
            self._ask(_sample_frame())
        self.assertIn("key", str(caught.exception).lower())

    def test_garbage_answer_raises_instead_of_guessing(self) -> None:
        self.server.responses = [(200, _chat_response("I think a bar chart would be nice!"))]  # type: ignore[attr-defined]
        with self.assertRaises(AdvisorError):
            self._ask(_sample_frame())

    def test_all_invalid_suggestions_raise(self) -> None:
        content = json.dumps({"charts": [{"kind": "histogram", "x": "segment"}]})
        self.server.responses = [(200, _chat_response(content))]  # type: ignore[attr-defined]
        with self.assertRaises(AdvisorError):
            self._ask(_sample_frame())


class AdvisorValidationTests(unittest.TestCase):
    def test_brief_contains_no_values(self) -> None:
        brief = dataset_brief(_sample_frame())
        text = json.dumps(brief)
        self.assertNotIn(SECRET, text)
        self.assertNotIn("4200", text)
        self.assertEqual(brief["rows"], 5)
        self.assertEqual(
            {column["name"] for column in brief["columns"]},
            {"segment", "monthly_spend", "signup_date"},
        )

    def test_non_localhost_http_endpoint_is_rejected(self) -> None:
        with self.assertRaises(AdvisorError):
            request_chart_advice(
                _sample_frame(),
                intent="Overview",
                base_url="http://example.com/v1",
                model="m",
            )


if __name__ == "__main__":
    unittest.main()
