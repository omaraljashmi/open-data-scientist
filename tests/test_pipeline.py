from __future__ import annotations

from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest

import pandas as pd
import requests

from ods import (
    AirtableDestination,
    PipelineConfig,
    PipelineError,
    WebhookDestination,
    apply_cleaning_actions,
    build_cleaning_recipe,
    cleaning_actions_from_recipe,
    dataframe_to_records,
    pipeline_config_from_json,
    pipeline_config_to_json,
    preview_payload,
    push_to_airtable,
    push_to_webhook,
    replay_cleaning_batches,
    suggest_cleaning_actions,
)
from scripts.run_pipeline import main as run_pipeline_main


BASE_ID = "appABCDEFGHIJKLMN"
NO_SLEEP = lambda seconds: None  # noqa: E731 - deliberate stub


class _StubHandler(BaseHTTPRequestHandler):
    def _serve(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        self.server.seen.append(  # type: ignore[attr-defined]
            {
                "method": self.command,
                "path": self.path,
                "headers": {key: value for key, value in self.headers.items()},
                "body": json.loads(body) if body else None,
            }
        )
        responses = self.server.responses  # type: ignore[attr-defined]
        status, payload = responses.pop(0) if responses else (200, {})
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    do_POST = _serve
    do_GET = _serve

    def log_message(self, *args: object) -> None:  # silence test output
        return


class _StubServerMixin(unittest.TestCase):
    def setUp(self) -> None:
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _StubHandler)
        self.server.seen = []  # type: ignore[attr-defined]
        self.server.responses = []  # type: ignore[attr-defined]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.session = requests.Session()
        self.session.trust_env = False  # never route test traffic through proxies

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


class RecordConversionTests(unittest.TestCase):
    def test_records_are_json_safe_and_missing_values_are_omitted(self) -> None:
        frame = pd.DataFrame(
            {
                "count": pd.array([1, 2], dtype="int64"),
                "price": [19.5, float("nan")],
                "active": [True, False],
                "seen": pd.to_datetime(["2026-06-01", "2026-06-02"]),
                "note": ["ok", None],
            }
        )
        records = dataframe_to_records(frame)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["count"], 1)
        self.assertIsInstance(records[0]["count"], int)
        self.assertEqual(records[0]["active"], True)
        self.assertTrue(records[0]["seen"].startswith("2026-06-01"))
        self.assertNotIn("price", records[1])
        self.assertNotIn("note", records[1])
        json.dumps(records)  # must be serializable end to end

    def test_row_limit_and_duplicate_columns(self) -> None:
        frame = pd.DataFrame({"a": [1, 2, 3]})
        self.assertEqual(len(dataframe_to_records(frame, row_limit=2)), 2)
        duplicated = pd.DataFrame([[1, 2]], columns=["a", "a"])
        with self.assertRaises(PipelineError):
            dataframe_to_records(duplicated)


class PipelineConfigTests(unittest.TestCase):
    def test_round_trip_contains_no_credential_fields(self) -> None:
        config = PipelineConfig(
            name="Nightly export",
            kind="airtable",
            airtable=AirtableDestination(base_id=BASE_ID, table="Customers"),
            row_limit=500,
        )
        text = pipeline_config_to_json(config)
        self.assertNotIn("token", text.lower())
        self.assertNotIn("secret", text.lower())
        self.assertEqual(pipeline_config_from_json(text), config)

    def test_unknown_fields_and_bad_kind_are_rejected(self) -> None:
        config = PipelineConfig(
            name="Hook",
            kind="webhook",
            webhook=WebhookDestination(url="https://example.com/ingest"),
        )
        payload = json.loads(pipeline_config_to_json(config))
        payload["surprise"] = True
        with self.assertRaises(PipelineError):
            pipeline_config_from_json(json.dumps(payload))
        with self.assertRaises(PipelineError):
            pipeline_config_from_json(json.dumps({"format": "ods-pipeline", "version": 1, "name": "x", "kind": "ftp"}))

    def test_webhook_requires_https_except_localhost(self) -> None:
        with self.assertRaises(PipelineError):
            pipeline_config_to_json(
                PipelineConfig(
                    name="Hook",
                    kind="webhook",
                    webhook=WebhookDestination(url="http://example.com/ingest"),
                )
            )
        pipeline_config_to_json(
            PipelineConfig(
                name="Hook",
                kind="webhook",
                webhook=WebhookDestination(url="http://localhost:9999/ingest"),
            )
        )


class AirtablePushTests(_StubServerMixin):
    def _destination(self) -> AirtableDestination:
        return AirtableDestination(base_id=BASE_ID, table="Customer Table")

    def test_batches_of_ten_with_auth_and_typecast(self) -> None:
        records = [{"n": index} for index in range(25)]
        report = push_to_airtable(
            records,
            self._destination(),
            "patTESTTOKEN",
            api_url=self.base_url,
            session=self.session,
            sleep=NO_SLEEP,
        )
        self.assertTrue(report.ok)
        self.assertEqual(report.sent_records, 25)
        self.assertEqual(report.batches_sent, 3)
        seen = self.server.seen  # type: ignore[attr-defined]
        self.assertEqual(len(seen), 3)
        self.assertEqual(seen[0]["path"], f"/{BASE_ID}/Customer%20Table")
        self.assertEqual(seen[0]["headers"]["Authorization"], "Bearer patTESTTOKEN")
        self.assertEqual(len(seen[0]["body"]["records"]), 10)
        self.assertEqual(len(seen[2]["body"]["records"]), 5)
        self.assertIs(seen[0]["body"]["typecast"], True)
        self.assertEqual(seen[0]["body"]["records"][0], {"fields": {"n": 0}})

    def test_auth_failure_stops_the_run_with_guidance(self) -> None:
        self.server.responses = [(401, {"error": {"type": "AUTHENTICATION_REQUIRED"}})]  # type: ignore[attr-defined]
        with self.assertRaises(PipelineError) as caught:
            push_to_airtable(
                [{"n": 1}],
                self._destination(),
                "patBAD",
                api_url=self.base_url,
                session=self.session,
                sleep=NO_SLEEP,
            )
        self.assertIn("token", str(caught.exception).lower())

    def test_field_rejection_is_recorded_and_run_continues(self) -> None:
        self.server.responses = [  # type: ignore[attr-defined]
            (200, {}),
            (422, {"error": {"type": "INVALID_VALUE_FOR_COLUMN", "message": "bad field"}}),
            (200, {}),
        ]
        records = [{"n": index} for index in range(25)]
        report = push_to_airtable(
            records,
            self._destination(),
            "patTESTTOKEN",
            api_url=self.base_url,
            session=self.session,
            sleep=NO_SLEEP,
        )
        self.assertFalse(report.ok)
        self.assertEqual(report.sent_records, 15)  # batches of 10 + 5; the 10-record batch 2 failed
        self.assertEqual(report.batches_sent, 2)
        self.assertEqual(len(report.failures), 1)
        self.assertIn("Batch 2", report.failures[0])

    def test_invalid_base_id_and_missing_token_fail_fast(self) -> None:
        with self.assertRaises(PipelineError):
            push_to_airtable([], AirtableDestination(base_id="nope", table="T"), "patX")
        with self.assertRaises(PipelineError):
            push_to_airtable([], self._destination(), "  ")


class WebhookPushTests(_StubServerMixin):
    def test_batches_carry_metadata_and_secret_header(self) -> None:
        destination = WebhookDestination(
            url=f"{self.base_url}/ingest".replace("127.0.0.1", "localhost"),
            batch_size=5,
            secret_header="X-ODS-Secret",
        )
        records = [{"n": index} for index in range(12)]
        report = push_to_webhook(
            records,
            destination,
            secret_value="shh",
            session=self.session,
            sleep=NO_SLEEP,
            pipeline_name="demo",
        )
        self.assertTrue(report.ok)
        seen = self.server.seen  # type: ignore[attr-defined]
        self.assertEqual(len(seen), 3)
        self.assertEqual(seen[0]["headers"]["X-ODS-Secret"], "shh")
        body = seen[0]["body"]
        self.assertEqual(body["source"], "open-data-scientist")
        self.assertEqual(body["pipeline"], "demo")
        self.assertEqual(body["batch_index"], 0)
        self.assertEqual(body["batch_count"], 3)
        self.assertEqual(len(body["records"]), 5)
        self.assertEqual(len(seen[2]["body"]["records"]), 2)

    def test_non_success_status_is_recorded(self) -> None:
        self.server.responses = [(500, {}), (200, {}), (200, {})]  # type: ignore[attr-defined]
        destination = WebhookDestination(url=f"{self.base_url}/ingest", batch_size=5)
        records = [{"n": index} for index in range(12)]
        report = push_to_webhook(
            records, destination, session=self.session, sleep=NO_SLEEP
        )
        self.assertFalse(report.ok)
        self.assertEqual(report.sent_records, 7)
        self.assertEqual(len(report.failures), 1)

    def test_missing_secret_value_fails_fast(self) -> None:
        destination = WebhookDestination(
            url=f"{self.base_url}/ingest", secret_header="X-ODS-Secret"
        )
        with self.assertRaises(PipelineError):
            push_to_webhook([{"n": 1}], destination, session=self.session, sleep=NO_SLEEP)


class CleaningRecipeReplayTests(unittest.TestCase):
    def test_recipe_replays_to_the_same_cleaned_frame(self) -> None:
        original = pd.DataFrame(
            {
                "name": [" Alice", "Bob ", "Bob ", "  ", "Cara"],
                "amount": ["1", "2", "2", "3", "4"],
            }
        )
        actions = suggest_cleaning_actions(original)
        self.assertTrue(actions)
        cleaned = apply_cleaning_actions(original, actions)

        recipe = build_cleaning_recipe(
            "sample.csv", "0" * 64, original, cleaned, [tuple(actions)]
        )
        replayed_batches = cleaning_actions_from_recipe(recipe)
        replayed = replay_cleaning_batches(original, replayed_batches)
        pd.testing.assert_frame_equal(replayed, cleaned)

    def test_invalid_recipes_are_rejected(self) -> None:
        from ods import CleaningError

        with self.assertRaises(CleaningError):
            cleaning_actions_from_recipe("not json")
        with self.assertRaises(CleaningError):
            cleaning_actions_from_recipe(json.dumps({"format": "other", "version": 1}))
        bad_operation = {
            "format": "open-data-scientist-cleaning-recipe",
            "version": 1,
            "operations": [{"operation": "explode", "confidence": "high"}],
        }
        with self.assertRaises(CleaningError):
            cleaning_actions_from_recipe(json.dumps(bad_operation))


class RunPipelineScriptTests(unittest.TestCase):
    def test_dry_run_prints_payload_and_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_path = Path(tmp) / "data.csv"
            data_path.write_text("a,b\n1,x\n2,y\n", encoding="utf-8")
            pipeline_path = Path(tmp) / "pipeline.json"
            pipeline_path.write_text(
                pipeline_config_to_json(
                    PipelineConfig(
                        name="Dry run",
                        kind="webhook",
                        webhook=WebhookDestination(url="http://localhost:9/ingest"),
                    )
                ),
                encoding="utf-8",
            )
            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = run_pipeline_main(
                    [
                        "--input",
                        str(data_path),
                        "--pipeline",
                        str(pipeline_path),
                        "--dry-run",
                    ]
                )
        self.assertEqual(exit_code, 0)
        self.assertIn('"records"', output.getvalue())


class PreviewPayloadTests(unittest.TestCase):
    def test_airtable_preview_matches_the_wire_format(self) -> None:
        config = PipelineConfig(
            name="Preview",
            kind="airtable",
            airtable=AirtableDestination(base_id=BASE_ID, table="T"),
        )
        payload = json.loads(preview_payload([{"a": 1}] * 15, config))
        self.assertEqual(len(payload["records"]), 10)
        self.assertIn("typecast", payload)


if __name__ == "__main__":
    unittest.main()
