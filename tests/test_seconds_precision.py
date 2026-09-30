"""秒级精度与边界口径测试。

覆盖：
* 30 秒等待必须向上取整到一分钟，建议时刻不得早于窗口末端；
* 分钟口径/秒口径下“恰好等于阈值”和“差一秒”的稳定分类；
* 展示分钟相同但业务判定不同（不得由展示舍入反推判定）；
* 两端同时受限时取恢复较晚的一端，左闭右开端点不受影响；
* 夏令时切换与跨午夜判断；
* 延长/恢复连续修改时原因链可追溯、状态翻转；
* 小数秒事件落库、重启与幂等重放后分类不漂移。
"""

from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from app.engine import chain_window, classify_flight
from app.models import (
    EVENT_CLOSED,
    IMPACT_CANCELLED,
    IMPACT_DELAYED,
    Airport,
    DisruptionEvent,
    Flight,
)
from app.timeutil import ceil_minutes, crosses_local_midnight, parse_event_datetime
from app.service import DisruptionService
from tests.support import ServiceTestCase
UTC = timezone.utc


def dt(value: str) -> datetime:
    return parse_event_datetime(value, "t")


def make_window(
    start: str = "2026-09-07T15:00:00Z",
    end: str | None = "2026-09-07T16:00:00Z",
    *,
    airport_code: str = "BSR",
    terminal: bool = False,
) -> object:
    event = DisruptionEvent(
        event_id="evt-seconds0001",
        event_version=1,
        event_type=EVENT_CLOSED,
        airport_code=airport_code,
        effective_from=dt(start),
        effective_until=dt(end) if end else None,
        reported_at=dt("2026-09-07T14:00:00Z"),
        supersedes_event_id=None,
        reason=None,
    )
    return chain_window(event, event, Airport(airport_code, "Test", "UTC", 0))


def make_flight(
    *,
    flight_id: str = "T-001-20260907",
    origin: str = "BSR",
    destination: str = "APS",
    departure: str = "2026-09-07T15:59:30Z",
    arrival: str = "2026-09-07T17:30:00Z",
    can_retime: bool = True,
    max_delay_minutes: int = 90,
    max_delay_seconds: int | None = None,
) -> Flight:
    return Flight(
        flight_id=flight_id,
        flight_number="T001",
        origin=origin,
        destination=destination,
        scheduled_departure=dt(departure),
        scheduled_arrival=dt(arrival),
        passenger_count=10,
        can_retime=can_retime,
        max_delay_minutes=max_delay_minutes,
        max_delay_seconds=max_delay_seconds,
    )


class SecondsRoundingTest(unittest.TestCase):
    def test_ceiling_helper(self) -> None:
        from datetime import timedelta

        self.assertEqual(ceil_minutes(timedelta(0)), 0)
        self.assertEqual(ceil_minutes(timedelta(seconds=1)), 1)
        self.assertEqual(ceil_minutes(timedelta(seconds=30)), 1)
        self.assertEqual(ceil_minutes(timedelta(seconds=60)), 1)
        self.assertEqual(ceil_minutes(timedelta(seconds=61)), 2)
        self.assertEqual(ceil_minutes(timedelta(seconds=3300.5)), 56)

    def test_thirty_second_hold_rounds_up_and_proposal_is_executable(self) -> None:
        # Scheduled 30 seconds before reopening: the old floor division yielded
        # a 0-minute "delay" and proposed the original (still-closed) time.
        flight = make_flight(departure="2026-09-07T15:59:30Z")
        record = classify_flight(flight, make_window())
        self.assertEqual(record["impact_status"], IMPACT_DELAYED)
        self.assertEqual(record["overlap_seconds"], 30)
        self.assertEqual(record["overlap_minutes"], 1)
        self.assertEqual(record["delay_minutes"], 1)
        self.assertEqual(record["delay_seconds"], 60)
        self.assertEqual(record["proposed_departure"], "2026-09-07T16:00:30Z")
        # The proposal must never land before the window end (16:00:00Z).
        proposed = parse_event_datetime(record["proposed_departure"], "p")
        self.assertGreaterEqual(proposed, dt("2026-09-07T16:00:00Z"))

    def test_one_second_hold_is_a_one_minute_executable_delay(self) -> None:
        flight = make_flight(departure="2026-09-07T15:59:59Z")
        record = classify_flight(flight, make_window())
        self.assertEqual(record["impact_status"], IMPACT_DELAYED)
        self.assertEqual(record["overlap_seconds"], 1)
        self.assertEqual(record["overlap_minutes"], 1)
        self.assertEqual(record["proposed_departure"], "2026-09-07T16:00:59Z")

    def test_exactly_at_window_end_is_not_affected_even_with_seconds(self) -> None:
        flight = make_flight(departure="2026-09-07T16:00:00Z")
        self.assertIsNone(classify_flight(flight, make_window()))
        # Half-open: one microsecond before the end is still inside.
        near_end = make_flight(departure="2026-09-07T15:59:59.999999Z")
        record = classify_flight(near_end, make_window(end="2026-09-07T16:00:00.000000Z"))
        self.assertEqual(record["overlap_seconds"], 0.000001)
        self.assertEqual(record["overlap_minutes"], 1)

    def test_exactly_at_minute_limit_is_delayed_one_second_over_cancels(self) -> None:
        window = make_window()  # ends 16:00:00
        at_limit = make_flight(
            departure="2026-09-07T15:59:00Z",  # wait 60s == 1 minute limit
            max_delay_minutes=1,
        )
        record = classify_flight(at_limit, window)
        self.assertEqual(record["impact_status"], IMPACT_DELAYED)
        self.assertEqual(record["delay_minutes"], 1)

        one_over = make_flight(
            flight_id="T-002-20260907",
            departure="2026-09-07T15:58:59Z",  # wait 61s -> ceiling 2 minutes
            max_delay_minutes=1,
        )
        record = classify_flight(one_over, window)
        self.assertEqual(record["impact_status"], IMPACT_CANCELLED)
        self.assertEqual(record["overlap_seconds"], 61)
        self.assertEqual(record["overlap_minutes"], 2)

    def test_second_precision_limit_equal_and_one_second_over(self) -> None:
        window = make_window()
        kw = dict(max_delay_minutes=999, max_delay_seconds=90)
        # 89.5s and 90.5s both display as 2 ceiling minutes; only the raw wait
        # decides, so display rounding cannot drive the business classification.
        under = make_flight(
            flight_id="T-010-20260907",
            departure="2026-09-07T15:58:30.5Z",  # wait 89.5s
            **kw,
        )
        equal = make_flight(
            flight_id="T-011-20260907",
            departure="2026-09-07T15:58:30Z",  # wait 90s exactly
            **kw,
        )
        over = make_flight(
            flight_id="T-012-20260907",
            departure="2026-09-07T15:58:29.5Z",  # wait 90.5s
            **kw,
        )
        r_under = classify_flight(under, window)
        r_equal = classify_flight(equal, window)
        r_over = classify_flight(over, window)
        self.assertEqual(r_under["impact_status"], IMPACT_DELAYED)
        self.assertEqual(r_equal["impact_status"], IMPACT_DELAYED)  # boundary inclusive
        self.assertEqual(r_over["impact_status"], IMPACT_CANCELLED)
        self.assertEqual(r_under["overlap_minutes"], r_over["overlap_minutes"])
        self.assertEqual(r_under["overlap_minutes"], 2)
        self.assertEqual(r_equal["overlap_seconds"], 90)
        # Executable hold for the exact 90s wait is ceiling 2 minutes; raw wait
        # and adopted operating time are both preserved.
        self.assertEqual(r_equal["delay_seconds"], 120)
        self.assertEqual(r_equal["delay_minutes"], 2)
        self.assertEqual(r_equal["proposed_departure"], "2026-09-07T16:00:30Z")

    def test_both_endpoints_constrained_uses_later_resuming_endpoint(self) -> None:
        # Same airport at both legs; departure waits 60s, arrival waits 30s.
        flight = make_flight(
            flight_id="T-020-20260907",
            origin="BSR",
            destination="BSR",
            departure="2026-09-07T15:59:00Z",
            arrival="2026-09-07T15:59:30Z",
            max_delay_minutes=1,
        )
        record = classify_flight(flight, make_window())
        self.assertEqual(record["affected_endpoint"], "both")
        self.assertEqual(record["overlap_seconds"], 60)
        self.assertEqual(record["impact_status"], IMPACT_DELAYED)
        # One more second at the origin wait pushes the max over the limit.
        flight = replace(flight, scheduled_departure=dt("2026-09-07T15:58:59Z"))
        record = classify_flight(flight, make_window())
        self.assertEqual(record["impact_status"], IMPACT_CANCELLED)
        self.assertEqual(record["overlap_seconds"], 61)


class DaylightSavingTest(unittest.TestCase):
    NY = ZoneInfo("America/New_York")

    def test_fall_back_window_waits_use_utc_elapsed_time(self) -> None:
        # Local 01:30 EDT (-04:00) -> 02:30 EST (-05:00) on 2026-11-01: the
        # clock reads a one-hour span but 2 physical hours elapse.
        start = datetime(2026, 11, 1, 5, 30, tzinfo=UTC)
        end = datetime(2026, 11, 1, 7, 30, tzinfo=UTC)
        self.assertFalse(crosses_local_midnight(start, end, self.NY))
        airport = Airport("JFK", "Test", "America/New_York", 0)
        event = DisruptionEvent(
            event_id="evt-dst0000001",
            event_version=1,
            event_type=EVENT_CLOSED,
            airport_code="JFK",
            effective_from=start,
            effective_until=end,
            reported_at=start,
            supersedes_event_id=None,
            reason=None,
        )
        window = chain_window(event, event, airport)
        flight = Flight(
            flight_id="D-001-20261101",
            flight_number="D001",
            origin="JFK",
            destination="LAX",
            scheduled_departure=start,
            scheduled_arrival=datetime(2026, 11, 1, 9, 0, tzinfo=UTC),
            passenger_count=1,
            can_retime=True,
            max_delay_minutes=180,
        )
        record = classify_flight(flight, window)
        self.assertEqual(record["overlap_seconds"], 7200)
        self.assertEqual(record["overlap_minutes"], 120)
        self.assertEqual(record["impact_status"], IMPACT_DELAYED)

    def test_spring_forward_window(self) -> None:
        # Local 01:30 -> 03:30 on 2026-03-08 spans one physical UTC hour.
        start = datetime(2026, 3, 8, 6, 30, tzinfo=UTC)
        end = datetime(2026, 3, 8, 7, 30, tzinfo=UTC)
        self.assertFalse(crosses_local_midnight(start, end, self.NY))

    def test_midnight_flag_stable_across_fall_back(self) -> None:
        # Local 23:30 EST? Build UTC bounds for local 2026-11-01 23:30 (EST,
        # -05:00) -> local 2026-11-02 01:00: two local dates, 90 clock minutes.
        start = datetime(2026, 11, 2, 4, 30, tzinfo=UTC)  # 23:30 EST on 11/1
        end = datetime(2026, 11, 2, 6, 0, tzinfo=UTC)  # 01:00 EST on 11/2
        self.assertTrue(crosses_local_midnight(start, end, self.NY))
        self.assertEqual(ceil_minutes(end - start), 90)


class SecondsChainServiceTest(ServiceTestCase):
    def close(self, **kw):
        payload = {
            "event_id": "evt-bsrsec-close",
            "event_version": 1,
            "event_type": "airport.closed",
            "airport_code": "BSR",
            "effective_from": "2026-09-07T15:00:00Z",
            "effective_until": "2026-09-07T16:35:00Z",
            "reported_at": "2026-09-07T14:00:00Z",
            "reason": "initial closure",
        }
        payload.update(kw)
        return payload

    def test_extend_one_second_flips_delayed_to_cancelled_and_reasons_trace(self) -> None:
        # BY205 departs BSR 15:05; window end 16:35 -> exactly its 90 minute
        # limit: boundary inclusive => delayed.
        first = self.service.submit_event(self.close())
        row = {i["flight_id"]: i for i in first["impacts"]}["BY-205-20260908"]
        self.assertEqual(row["impact_status"], IMPACT_DELAYED)
        self.assertEqual(row["overlap_seconds"], 5400)
        self.assertEqual(row["delay_minutes"], 90)
        self.assertEqual(row["proposed_departure"], "2026-09-07T16:35:00Z")

        # Extend the end by one second: ceiling 91 minutes -> cancelled.
        extended = self.service.submit_event(
            {
                "event_id": "evt-bsrsec-extend",
                "event_version": 2,
                "event_type": "airport.extended",
                "airport_code": "BSR",
                "effective_from": "2026-09-07T16:20:00Z",
                "effective_until": "2026-09-07T16:35:01Z",
                "reported_at": "2026-09-07T16:31:00Z",
                "supersedes_event_id": "evt-bsrsec-close",
                "reason": "one-second operational extension",
            }
        )
        row = {i["flight_id"]: i for i in extended["impacts"]}["BY-205-20260908"]
        self.assertEqual(row["impact_status"], IMPACT_CANCELLED)
        self.assertEqual(row["overlap_seconds"], 5401)
        self.assertEqual(row["overlap_minutes"], 91)
        self.assertIsNone(row["delay_minutes"])

        # Reopen early enough that the 15 minute buffer ends exactly at the
        # 15:05 departure: touching the half-open end frees the flight.
        reopened = self.service.submit_event(
            {
                "event_id": "evt-bsrsec-reopen",
                "event_version": 3,
                "event_type": "airport.reopened",
                "airport_code": "BSR",
                "effective_from": "2026-09-07T14:50:00Z",
                "reported_at": "2026-09-07T14:55:00Z",
                "supersedes_event_id": "evt-bsrsec-extend",
                "reason": "runway reopened early",
            }
        )
        self.assertEqual(reopened["impact_count"], 0)
        self.assertEqual(reopened["resolved_count"], 1)

        # Every snapshot is attributable to its own event and stated reason.
        by_event = {
            "evt-bsrsec-close": ("initial closure", IMPACT_DELAYED),
            "evt-bsrsec-extend": ("one-second operational extension", IMPACT_CANCELLED),
            "evt-bsrsec-reopen": ("runway reopened early", None),
        }
        for event_id, (reason, status) in by_event.items():
            detail = self.service.event_status(event_id)
            self.assertEqual(detail["event"]["reason"], reason)
            if status is not None:
                rows = [i for i in detail["impacts"] if i["flight_id"] == "BY-205-20260908"]
                self.assertEqual(rows[0]["impact_status"], status)

        # The resolved flight must not linger in the latest-snapshot view.
        page = self.service.affected_flights(
            airport="BSR", status=None, limit=50, offset=0
        )
        self.assertNotIn("BY-205-20260908", [f["flight_id"] for f in page["flights"]])


class FixtureLoadedSecondLimitTest(ServiceTestCase):
    def test_loaded_flight_with_second_precision_airline_limit(self) -> None:
        from app.engine import chain_window
        from app.timeutil import parse_event_datetime

        # A window whose end sits 55 minutes + 30 seconds after BY205's
        # departure at BSR. Its fixture minute limit is 90, but override the
        # airline limit to 55 minutes 29 seconds: display ceiling is 56 minutes
        # under both readings, yet the raw-second limit must reject the hold.
        flight = replace(self.flights["BY-205-20260908"], max_delay_seconds=55 * 60 + 29)
        event = DisruptionEvent(
            event_id="evt-fixture-sec1",
            event_version=1,
            event_type=EVENT_CLOSED,
            airport_code="BSR",
            effective_from=dt("2026-09-07T15:00:00Z"),
            effective_until=dt("2026-09-07T16:00:30Z"),
            reported_at=dt("2026-09-07T14:00:00Z"),
            supersedes_event_id=None,
            reason=None,
        )
        window = chain_window(event, event, self.airports["BSR"])
        record = classify_flight(flight, window)
        self.assertEqual(record["overlap_seconds"], 3330)  # 55 min 30 s raw
        self.assertEqual(record["overlap_minutes"], 56)
        self.assertEqual(record["impact_status"], IMPACT_CANCELLED)

        # Raising the limit by one second makes the same hold retimable.
        flight = replace(flight, max_delay_seconds=55 * 60 + 30)
        record = classify_flight(flight, window)
        self.assertEqual(record["impact_status"], IMPACT_DELAYED)
        self.assertEqual(record["delay_minutes"], 56)
        self.assertEqual(record["delay_seconds"], 3360)
        self.assertEqual(record["proposed_departure"], "2026-09-07T16:01:00Z")

    def test_loader_accepts_second_precision_limit_from_fixture(self) -> None:
        import json
        import tempfile
        from pathlib import Path

        from app.config import load_flights

        item = {
            "flight_id": "Z-900-20260907",
            "flight_number": "Z900",
            "origin": "BSR",
            "destination": "APS",
            "scheduled_departure": "2026-09-07T15:05:00Z",
            "scheduled_arrival": "2026-09-07T16:45:00Z",
            "passenger_count": 1,
            "can_retime": True,
            "max_delay_minutes": 90,
            "max_delay_seconds": 5400,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "flights.json"
            path.write_text(json.dumps([item]), encoding="utf-8")
            loaded = load_flights(Path(tmp), self.airports)
        self.assertEqual(loaded["Z-900-20260907"].max_delay_seconds, 5400)

        # A missing field keeps the legacy minute-only semantics.
        item.pop("max_delay_seconds")
        item["flight_id"] = "Z-901-20260907"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "flights.json"
            path.write_text(json.dumps([item]), encoding="utf-8")
            loaded = load_flights(Path(tmp), self.airports)
        self.assertIsNone(loaded["Z-901-20260907"].max_delay_seconds)


class FractionalSecondPersistenceTest(ServiceTestCase):
    def test_fractional_window_classification_survives_restart_and_replay(self) -> None:
        # BSR window ends half a second after the minute. The old floor rule
        # proposed 16:00:00 - before the airport actually reopens at 16:00:00.5.
        payload = {
            "event_id": "evt-frac0000001",
            "event_version": 1,
            "event_type": "airport.closed",
            "airport_code": "BSR",
            "effective_from": "2026-09-07T15:00:00Z",
            "effective_until": "2026-09-07T16:00:00.5Z",
            "reported_at": "2026-09-07T14:00:00.25Z",
        }
        first = self.service.submit_event(payload)
        row = {i["flight_id"]: i for i in first["impacts"]}["BY-205-20260908"]
        self.assertEqual(row["impact_status"], IMPACT_DELAYED)
        self.assertEqual(row["overlap_seconds"], 3300.5)  # raw wait preserved
        self.assertEqual(row["overlap_minutes"], 56)  # ceiling, not the old 55
        self.assertEqual(row["delay_seconds"], 3360)  # adopted hold
        self.assertEqual(row["proposed_departure"], "2026-09-07T16:01:00Z")
        self.assertEqual(row["proposed_arrival"], "2026-09-07T17:41:00Z")

        # In-process idempotent replay keeps the exact same result.
        replay = self.service.submit_event(dict(payload))
        self.assertEqual(replay["processing_state"], "replayed")
        self.assertEqual(replay["impacts"], first["impacts"])

        # Restart: stored fractional precision must not reclassify the flight.
        self.restart_service()
        status = self.service.event_status(payload["event_id"])
        self.assertEqual(status["impacts"], first["impacts"])
        self.assertEqual(status["event"]["effective_until"], "2026-09-07T16:00:00.500000Z")
        replay_after = self.service.submit_event(dict(payload))
        self.assertEqual(replay_after["processing_state"], "replayed")
        self.assertEqual(replay_after["impacts"], first["impacts"])

    def test_fractional_start_window_survives_restart(self) -> None:
        payload = {
            "event_id": "evt-frac0000002",
            "event_version": 1,
            "event_type": "airport.closed",
            "airport_code": "APS",
            "effective_from": "2026-09-07T15:00:00.5Z",
            "effective_until": "2026-09-07T19:00:00Z",
            "reported_at": "2026-09-07T14:00:00Z",
        }
        first = self.service.submit_event(payload)
        by_flight = {i["flight_id"]: i for i in first["impacts"]}
        self.assertEqual(
            {f: i["impact_status"] for f, i in by_flight.items()},
            {
                "AX-410-20260907": IMPACT_CANCELLED,
                "AX-412-20260908": IMPACT_CANCELLED,
                "BY-205-20260908": IMPACT_CANCELLED,
            },
        )
        # Waits are measured from the schedule to the window end, independent
        # of the fractional start; the fractional start itself must persist.
        self.assertEqual(by_flight["AX-410-20260907"]["overlap_seconds"], 12600)
        self.assertEqual(by_flight["AX-410-20260907"]["overlap_minutes"], 210)
        self.restart_service()
        status = self.service.event_status(payload["event_id"])
        self.assertEqual(status["event"]["effective_from"], "2026-09-07T15:00:00.500000Z")
        self.assertEqual(status["impacts"], first["impacts"])


class LegacySchemaMigrationTest(ServiceTestCase):
    def test_legacy_database_without_seconds_columns_migrates(self) -> None:
        import sqlite3

        from app.repository import Repository

        self.repo.close()
        # Remove the fresh-schema database so the legacy DDL below starts clean.
        for suffix in ("", "-wal", "-shm"):
            path = self.db_path.with_name(self.db_path.name + suffix)
            if path.exists():
                path.unlink()
        # Build a database file in the pre-seconds schema with a legacy row.
        conn = sqlite3.connect(self.db_path)
        conn.executescript(
            """
            CREATE TABLE events (
                event_id TEXT PRIMARY KEY, event_version INTEGER NOT NULL,
                event_type TEXT NOT NULL, airport_code TEXT NOT NULL,
                effective_from TEXT NOT NULL, effective_until TEXT,
                reported_at TEXT NOT NULL, supersedes_event_id TEXT,
                reason TEXT, payload_json TEXT NOT NULL,
                replay_count INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL
            );
            CREATE TABLE impacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL, root_event_id TEXT NOT NULL,
                airport_code TEXT NOT NULL, flight_id TEXT NOT NULL,
                flight_number TEXT NOT NULL, affected_endpoint TEXT NOT NULL,
                impact_status TEXT NOT NULL, overlap_minutes INTEGER,
                delay_minutes INTEGER, proposed_departure TEXT,
                proposed_arrival TEXT, passenger_count INTEGER NOT NULL,
                crosses_midnight INTEGER NOT NULL
            );
            """
        )
        conn.execute(
            "INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?,?,0,?)",
            (
                "evt-legacy00001", 1, "airport.closed", "BSR",
                "2026-09-07T15:00:00Z", "2026-09-07T16:00:00Z",
                "2026-09-07T14:00:00Z", None, None, "{}", "2026-09-07T14:00:00Z",
            ),
        )
        conn.execute(
            "INSERT INTO impacts (event_id, root_event_id, airport_code, flight_id, "
            "flight_number, affected_endpoint, impact_status, overlap_minutes, "
            "delay_minutes, proposed_departure, proposed_arrival, passenger_count, "
            "crosses_midnight) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "evt-legacy00001", "evt-legacy00001", "BSR", "BY-205-20260908",
                "BY205", "origin", "delayed", 55, 55,
                "2026-09-07T16:00:00Z", "2026-09-07T17:40:00Z", 131, 0,
            ),
        )
        conn.commit()
        conn.close()

        # Reopening the file must migrate; legacy rows keep their minute values
        # and simply have no second-precision figure.
        self.repo = Repository(self.db_path)
        self.service = DisruptionService(self.repo, self.airports, self.flights)
        status = self.service.event_status("evt-legacy00001")
        legacy = status["impacts"][0]
        self.assertEqual(legacy["overlap_minutes"], 55)
        self.assertIsNone(legacy["overlap_seconds"])

        # New events written after migration carry full second precision.
        result = self.service.submit_event(
            {
                "event_id": "evt-postmig0001",
                "event_version": 1,
                "event_type": "airport.closed",
                "airport_code": "KTA",
                "effective_from": "2026-09-07T16:30:00Z",
                "effective_until": "2026-09-07T18:00:00Z",
                "reported_at": "2026-09-07T14:00:00Z",
            }
        )
        kx = {i["flight_id"]: i for i in result["impacts"]}["KX-099-20260908"]
        self.assertEqual(kx["overlap_seconds"], 1200)
        self.assertEqual(kx["overlap_minutes"], 20)


if __name__ == "__main__":
    unittest.main()
