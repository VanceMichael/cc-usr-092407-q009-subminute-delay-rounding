"""秒级边界、阈值、夏令时、双端点与重启稳定性测试。

这些测试守护以下口径：

* 左闭右开端点：恰落在窗口末端不受影响，末端前 1 秒也要等待。
* 任何正的等待秒数都向上取整到可执行分钟，建议时刻不早于开放时刻。
* 最大可延误阈值按精确秒判定：恰好等于阈值可改时，差一秒越过阈值即取消。
* 双端点同时受限时取最大的精确等待秒，再统一取整。
* 夏令时切换夜，等待按 UTC 真实经过秒数计算，跨午夜按机场本地日历日判定。
* 原始等待秒数、采用的分钟与建议时刻一并落库，重启后分类不变。
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from app.engine import chain_window, classify_flight, airport_tz
from app.errors import ValidationError
from app.models import (
    EVENT_CLOSED,
    IMPACT_CANCELLED,
    IMPACT_DELAYED,
    Airport,
    DisruptionEvent,
    Flight,
)
from app.timeutil import (
    ceil_executable_minutes,
    crosses_local_midnight,
    wait_seconds,
)
from app.service import DisruptionService
from tests.support import ServiceTestCase


def z(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)


# --------------------------------------------------------------------------- #
# Pure rounding helpers
# --------------------------------------------------------------------------- #


class CeilingRoundingTest(unittest.TestCase):
    def test_zero_wait_is_zero_minutes(self) -> None:
        self.assertEqual(ceil_executable_minutes(0), 0)

    def test_any_positive_wait_rounds_up_to_a_whole_minute(self) -> None:
        for secs in (1, 30, 59, 60):
            self.assertEqual(ceil_executable_minutes(secs), 1, secs)
        self.assertEqual(ceil_executable_minutes(61), 2)
        self.assertEqual(ceil_executable_minutes(3600), 60)

    def test_wait_seconds_is_utc_elapsed_time(self) -> None:
        self.assertEqual(wait_seconds(z("2026-09-07T15:05:00Z"), z("2026-09-07T15:05:30Z")), 30)
        self.assertEqual(wait_seconds(z("2026-09-07T15:05:00Z"), z("2026-09-07T16:35:00Z")), 5400)


# --------------------------------------------------------------------------- #
# End-to-end service: second-level boundaries, persistence, chain traceability
# --------------------------------------------------------------------------- #


def bsr_window(event_id: str, until: str, **kw) -> dict:
    payload = {
        "event_id": event_id,
        "event_version": 1,
        "event_type": "airport.closed",
        "airport_code": "BSR",
        # long enough to clear the 15-minute minimum window for every scenario
        "effective_from": kw.pop("effective_from", "2026-09-07T14:00:00Z"),
        "effective_until": until,
        "reported_at": "2026-09-07T13:00:00Z",
    }
    payload.update(kw)
    return payload


class SecondBoundaryServiceTest(ServiceTestCase):
    BY = "BY-205-20260908"  # departs BSR 15:05:00Z, max_delay 90 min, can retime

    def _by_impact(self, result: dict) -> dict:
        records = {i["flight_id"]: i for i in result["impacts"]}
        self.assertIn(self.BY, records)
        return records[self.BY]

    def test_thirty_second_hold_rounds_up_and_proposes_executable_time(self) -> None:
        # Airport reopens at 15:05:30Z; the 15:05:00 scheduled departure must
        # hold 30 real seconds. Floor rounding used to report 0 minutes and an
        # unchanged (non-executable) schedule.
        result = self.service.submit_event(
            bsr_window("evt-secs-30-0001", "2026-09-07T15:05:30Z")
        )
        impact = self._by_impact(result)
        self.assertEqual(impact["impact_status"], IMPACT_DELAYED)
        self.assertEqual(impact["wait_seconds"], 30)
        self.assertEqual(impact["overlap_minutes"], 1)
        self.assertEqual(impact["delay_minutes"], 1)
        # Adopted operational time is the whole-minute ceiling and is at/after
        # the actual reopening instant.
        self.assertEqual(impact["proposed_departure"], "2026-09-07T15:06:00Z")
        self.assertGreaterEqual(
            z(impact["proposed_departure"]), z("2026-09-07T15:05:30Z")
        )
        self.assertEqual(impact["proposed_arrival"], "2026-09-07T16:46:00Z")

    def test_one_second_before_endpoint_is_affected(self) -> None:
        # Exactly a 15-minute window [14:50:01, 15:05:01); departure is 1 s
        # before the half-open end.
        result = self.service.submit_event(
            bsr_window(
                "evt-secs-1-00001",
                "2026-09-07T15:05:01Z",
                effective_from="2026-09-07T14:50:01Z",
            )
        )
        impact = self._by_impact(result)
        self.assertEqual(impact["impact_status"], IMPACT_DELAYED)
        self.assertEqual(impact["wait_seconds"], 1)
        self.assertEqual(impact["overlap_minutes"], 1)
        self.assertEqual(impact["proposed_departure"], "2026-09-07T15:06:00Z")

    def test_touching_endpoint_is_not_affected_at_second_granularity(self) -> None:
        # Window ends exactly on the scheduled minute; half-open interval frees
        # the flight even though inputs now carry seconds.
        result = self.service.submit_event(
            bsr_window(
                "evt-secs-touch-01",
                "2026-09-07T15:05:00Z",
                effective_from="2026-09-07T14:50:00Z",
            )
        )
        self.assertEqual(result["impact_count"], 0)

    def test_required_delay_exactly_at_threshold_is_delayed(self) -> None:
        # wait = 90:00 exactly -> adopted 90 <= max 90 -> delayed
        result = self.service.submit_event(
            bsr_window("evt-secs-exact-01", "2026-09-07T16:35:00Z")
        )
        impact = self._by_impact(result)
        self.assertEqual(impact["impact_status"], IMPACT_DELAYED)
        self.assertEqual(impact["wait_seconds"], 90 * 60)
        self.assertEqual(impact["delay_minutes"], 90)

    def test_one_second_over_threshold_cancels(self) -> None:
        # wait = 90:01 -> ceiling 91 > 90 -> cancelled
        result = self.service.submit_event(
            bsr_window("evt-secs-over-001", "2026-09-07T16:35:01Z")
        )
        impact = self._by_impact(result)
        self.assertEqual(impact["impact_status"], IMPACT_CANCELLED)
        self.assertEqual(impact["wait_seconds"], 90 * 60 + 1)
        self.assertEqual(impact["overlap_minutes"], 91)
        self.assertIsNone(impact["delay_minutes"])

    def test_one_second_under_threshold_still_delayed(self) -> None:
        # wait = 89:59 -> ceiling to 90, still <= 90 -> delayed
        result = self.service.submit_event(
            bsr_window("evt-secs-under-01", "2026-09-07T16:34:59Z")
        )
        impact = self._by_impact(result)
        self.assertEqual(impact["impact_status"], IMPACT_DELAYED)
        self.assertEqual(impact["wait_seconds"], 90 * 60 - 1)
        self.assertEqual(impact["delay_minutes"], 90)

    def test_fractional_seconds_are_rejected(self) -> None:
        with self.assertRaises(ValidationError) as ctx:
            self.service.submit_event(
                bsr_window("evt-secs-frac-01", "2026-09-07T15:05:30.5Z")
            )
        self.assertIn("whole-second", str(ctx.exception.details))

    def test_minimum_window_exact_boundary_is_second_precise(self) -> None:
        # Exactly 15:00 long is accepted; one second shorter is rejected.
        ok = self.service.submit_event(
            bsr_window(
                "evt-win-exact-01",
                "2026-09-07T14:15:00Z",
                effective_from="2026-09-07T14:00:00Z",
            )
        )
        self.assertEqual(ok["processing_state"], "processed")
        with self.assertRaises(ValidationError) as ctx:
            self.service.submit_event(
                bsr_window(
                    "evt-win-short-01",
                    "2026-09-07T14:14:59Z",
                    effective_from="2026-09-07T14:00:00Z",
                )
            )
        self.assertIn("window_too_short", str(ctx.exception.details))
        self.assertIn("window_seconds", str(ctx.exception.details))

    def test_classification_survives_restart_without_rounding_drift(self) -> None:
        payload = bsr_window("evt-secs-restart1", "2026-09-07T15:05:30Z")
        first = self.service.submit_event(payload)
        first_impact = self._by_impact(first)
        self.restart_service()
        status = self.service.event_status(payload["event_id"])
        restarted = {i["flight_id"]: i for i in status["impacts"]}[self.BY]
        self.assertEqual(restarted, first_impact)
        self.assertEqual(restarted["wait_seconds"], 30)
        self.assertEqual(restarted["delay_minutes"], 1)
        self.assertEqual(restarted["proposed_departure"], "2026-09-07T15:06:00Z")
        # Idempotent replay against the persisted row stays identical too.
        replay = self.service.submit_event(dict(payload))
        self.assertEqual(self._by_impact(replay), first_impact)

    def test_extend_and_reopen_keep_reason_traceable_via_root(self) -> None:
        close = {
            "event_id": "evt-chain-root-01",
            "event_version": 1,
            "event_type": "airport.closed",
            "airport_code": "APS",
            "effective_from": "2026-09-07T16:20:00Z",
            "effective_until": "2026-09-07T17:00:00Z",
            "reported_at": "2026-09-07T16:00:00Z",
            "reason": "thunderstorm cell over field",
        }
        first = self.service.submit_event(close)
        by_flight = {i["flight_id"]: i for i in first["impacts"]}
        self.assertEqual(by_flight["BY-205-20260908"]["impact_status"], IMPACT_DELAYED)
        self.assertTrue(
            all(i["root_event_id"] == "evt-chain-root-01" for i in first["impacts"])
        )

        extended = self.service.submit_event(
            {
                "event_id": "evt-chain-ext-001",
                "event_version": 2,
                "event_type": "airport.extended",
                "airport_code": "APS",
                "effective_from": "2026-09-07T16:55:00Z",
                "effective_until": "2026-09-07T19:30:00Z",
                "reported_at": "2026-09-07T16:40:00Z",
                "supersedes_event_id": "evt-chain-root-01",
                "reason": "storm tracking slower than forecast",
            }
        )
        ext = {i["flight_id"]: i for i in extended["impacts"]}
        # The earlier delayed flight flips to cancelled; every new snapshot still
        # points at the originating closure so the cause stays traceable.
        self.assertEqual(ext["BY-205-20260908"]["impact_status"], IMPACT_CANCELLED)
        self.assertTrue(
            all(i["root_event_id"] == "evt-chain-root-01" for i in extended["impacts"])
        )

        reopened = self.service.submit_event(
            {
                "event_id": "evt-chain-reopen1",
                "event_version": 3,
                "event_type": "airport.reopened",
                "airport_code": "APS",
                "effective_from": "2026-09-07T16:25:00Z",
                "reported_at": "2026-09-07T16:30:00Z",
                "supersedes_event_id": "evt-chain-ext-001",
            }
        )
        # AX412 stays cancelled; BY205 (arrives exactly at resume time) and
        # KX099 are freed. The reopen snapshot's active rows plus its resolved
        # tombstones all reference the same originating root chain.
        self.assertEqual(
            [i["flight_id"] for i in reopened["impacts"]], ["AX-412-20260908"]
        )
        self.assertEqual(reopened["resolved_count"], 2)
        rows = self.repo.get_impacts("evt-chain-reopen1")
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(r["root_event_id"] == "evt-chain-root-01" for r in rows))
        # The originating reason is preserved verbatim on the root event.
        root_status = self.service.event_status("evt-chain-root-01")
        self.assertEqual(root_status["event"]["reason"], "thunderstorm cell over field")


# --------------------------------------------------------------------------- #
# Legacy database migration
# --------------------------------------------------------------------------- #


class LegacySchemaMigrationTest(ServiceTestCase):
    def test_old_database_without_wait_seconds_is_backfilled(self) -> None:
        import sqlite3

        from app.repository import Repository

        # A whole-minute scenario: BSR 15:00-16:00, BY205 departs 15:05 ->
        # 55 minute hold, delayed, proposed 16:00. That is exactly what a
        # pre-fix build (whole-second inputs, floor rounding) would store.
        payload = bsr_window(
            "evt-migrate-001",
            "2026-09-07T16:00:00Z",
            effective_from="2026-09-07T15:00:00Z",
        )
        first = self.service.submit_event(payload)
        first_impact = {i["flight_id"]: i for i in first["impacts"]}[
            "BY-205-20260908"
        ]
        self.assertEqual(first_impact["wait_seconds"], 55 * 60)

        # Rebuild the impacts table as the pre-fix schema (drop wait_seconds)
        # to simulate a database file written by an older build, then reopen it.
        self.repo.close()
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cols = [r[1] for r in conn.execute("PRAGMA table_info(impacts)").fetchall()]
        old_cols = [c for c in cols if c != "wait_seconds"]
        rows = conn.execute(
            f"SELECT {', '.join(old_cols)} FROM impacts"
        ).fetchall()
        conn.execute("ALTER TABLE impacts RENAME TO impacts_old")
        conn.execute(
            """
            CREATE TABLE impacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL REFERENCES events(event_id),
                root_event_id TEXT NOT NULL, airport_code TEXT NOT NULL,
                flight_id TEXT NOT NULL, flight_number TEXT NOT NULL,
                affected_endpoint TEXT NOT NULL, impact_status TEXT NOT NULL,
                overlap_minutes INTEGER, delay_minutes INTEGER,
                proposed_departure TEXT, proposed_arrival TEXT,
                passenger_count INTEGER NOT NULL, crosses_midnight INTEGER NOT NULL,
                UNIQUE(event_id, flight_id, airport_code)
            )
            """
        )
        placeholders = ", ".join(f":{c}" for c in old_cols)
        conn.executemany(
            f"INSERT INTO impacts ({', '.join(old_cols)}) "
            f"VALUES ({placeholders})",
            [dict(r) for r in rows],
        )
        conn.execute("DROP TABLE impacts_old")
        conn.commit()
        conn.close()

        # Reopening runs the additive migration: column added, precise seconds
        # reconstructed from the adopted whole minutes, classification untouched.
        self.repo = Repository(self.db_path)
        self.service = DisruptionService(self.repo, self.airports, self.flights)
        status = self.service.event_status(payload["event_id"])
        migrated = {i["flight_id"]: i for i in status["impacts"]}[
            "BY-205-20260908"
        ]
        self.assertEqual(migrated["impact_status"], IMPACT_DELAYED)
        self.assertEqual(migrated["overlap_minutes"], 55)
        self.assertEqual(migrated["delay_minutes"], 55)
        self.assertEqual(migrated["wait_seconds"], 55 * 60)
        self.assertEqual(migrated["proposed_departure"], "2026-09-07T16:00:00Z")
        # Idempotent replay against the migrated row is still stable.
        replay = self.service.submit_event(dict(payload))
        self.assertEqual(
            {i["flight_id"]: i for i in replay["impacts"]}[
                "BY-205-20260908"
            ],
            migrated,
        )


# --------------------------------------------------------------------------- #
# Both endpoints constrained + daylight-saving boundary (engine level)
# --------------------------------------------------------------------------- #


class BothEndpointsAndDstTest(unittest.TestCase):
    def _dst_airport(self) -> Airport:
        return Airport("DST", "DST Field", "America/New_York", 0)

    def _event(self, **kw) -> DisruptionEvent:
        defaults = dict(
            event_id="evt-dst-0000001",
            event_version=1,
            event_type=EVENT_CLOSED,
            airport_code="DST",
            effective_from=z("2026-03-08T03:30:00Z"),
            effective_until=z("2026-03-08T07:30:00Z"),
            reported_at=z("2026-03-08T03:00:00Z"),
            supersedes_event_id=None,
            reason=None,
        )
        defaults.update(kw)
        return DisruptionEvent(**defaults)

    def test_spring_dst_waits_utc_seconds_and_flags_local_midnight(self) -> None:
        # 2026-03-08 in America/New_York: clocks jump 02:00 -> 03:00 at 07:00Z.
        # The window spans 03:30Z-07:30Z, i.e. local 22:30 (Mar 7 EST) -> 03:30
        # (Mar 8 EDT): it crosses local midnight even though the clock reading
        # skips an hour. The operational wait must be four real UTC hours.
        airport = self._dst_airport()
        event = self._event()
        window = chain_window(event, event, airport)
        flight = Flight(
            flight_id="DST-1",
            flight_number="DST1",
            origin="DST",
            destination="ZZZ",
            scheduled_departure=z("2026-03-08T03:30:00Z"),  # == window.start
            scheduled_arrival=z("2026-03-08T09:00:00Z"),
            passenger_count=1,
            can_retime=True,
            max_delay_minutes=300,
        )
        record = classify_flight(flight, window)
        self.assertIsNotNone(record)
        self.assertEqual(record["wait_seconds"], 4 * 3600)
        self.assertEqual(record["overlap_minutes"], 240)
        self.assertEqual(record["impact_status"], IMPACT_DELAYED)
        self.assertEqual(
            record["proposed_departure"], "2026-03-08T07:30:00Z"
        )
        self.assertTrue(
            crosses_local_midnight(window.start, window.end, airport_tz(airport))
        )

    def test_both_endpoints_take_max_precise_wait(self) -> None:
        airport = self._dst_airport()
        event = self._event()
        window = chain_window(event, event, airport)
        # Origin departs at 03:30Z (waits 4h), destination arrives at 07:00Z
        # (waits 30 min) at the same constrained airport.
        flight = Flight(
            flight_id="DST-2",
            flight_number="DST2",
            origin="DST",
            destination="DST",
            scheduled_departure=z("2026-03-08T03:30:00Z"),
            scheduled_arrival=z("2026-03-08T07:00:00Z"),
            passenger_count=2,
            can_retime=True,
            max_delay_minutes=300,
        )
        record = classify_flight(flight, window)
        self.assertEqual(record["affected_endpoint"], "both")
        # 4h origin wait dominates the 30-minute destination wait.
        self.assertEqual(record["wait_seconds"], 4 * 3600)
        self.assertEqual(record["overlap_minutes"], 240)
        self.assertEqual(record["impact_status"], IMPACT_DELAYED)
        # Both proposed times shift by the same adopted minutes.
        self.assertEqual(record["proposed_departure"], "2026-03-08T07:30:00Z")
        self.assertEqual(record["proposed_arrival"], "2026-03-08T11:00:00Z")

    def test_fall_back_dst_local_midnight_is_stable(self) -> None:
        # 2026-11-01 in America/New_York: clocks fall back 02:00 -> 01:00 at
        # 06:00Z. Window 04:30Z-07:00Z is local 00:30 (EST, Nov 1) -> 02:00
        # (EST, Nov 1): single local date despite the repeated hour.
        ny = ZoneInfo("America/New_York")
        self.assertFalse(
            crosses_local_midnight(
                z("2026-11-01T04:30:00Z"), z("2026-11-01T07:00:00Z"), ny
            )
        )
        # Starting before local midnight (03:30Z == 23:30 EDT Oct 31) crosses.
        self.assertTrue(
            crosses_local_midnight(
                z("2026-11-01T03:30:00Z"), z("2026-11-01T07:00:00Z"), ny
            )
        )
        # The precise wait across the repeated hour is 3.5 real UTC hours.
        self.assertEqual(
            wait_seconds(z("2026-11-01T03:30:00Z"), z("2026-11-01T07:00:00Z")),
            int(timedelta(hours=3, minutes=30).total_seconds()),
        )


if __name__ == "__main__":
    unittest.main()
