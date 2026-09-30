"""航班影响计算引擎。

规则：

* 比较前将所有时间统一转换为带时区的 UTC。时间窗口采用左闭右开语义，
  端点相接不算重叠。
* 航班从受影响机场起飞或抵达受影响机场的计划时刻落入关闭窗口时受影响。
* ``airport.closed`` 建立事件链，``effective_until = null`` 表示结束时间未知。
* ``airport.extended`` 延续事件链，并把窗口延长到当前事件的结束时刻。
* ``airport.reopened`` 结束事件链，机场在恢复缓冲时间结束后重新运行。
* 结束时间未知时结果为 ``pending_confirmation``；可在最大延误内改时的结果
  为 ``delayed``；其余受影响航班为 ``cancelled``。

秒级口径：

* 业务判定基于精确的等待*秒数*：先在所有受影响端点间取最大等待秒数，再
  一次性向上取整到可执行分钟。任何正的等待，哪怕不足一分钟，也要占一个
  可执行分钟，建议时刻绝不早于机场实际开放时刻。
* ``wait_seconds`` 保留原始等待秒数，``delay_minutes``/``overlap_minutes``
  是最终采用的分钟口径，建议运行时刻按采用的分钟生成。展示用分钟数不得
  反过来决定业务分类。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from app.models import (
    EVENT_CLOSED,
    EVENT_EXTENDED,
    EVENT_REOPENED,
    Airport,
    DisruptionEvent,
    Flight,
    IMPACT_CANCELLED,
    IMPACT_DELAYED,
    IMPACT_PENDING,
)
from app.timeutil import ceil_executable_minutes, crosses_local_midnight, wait_seconds
from app.models import iso_utc

# Severity ordering used when one flight is affected at both endpoints.
_SEVERITY = {IMPACT_DELAYED: 0, IMPACT_PENDING: 1, IMPACT_CANCELLED: 2}


@dataclass(frozen=True)
class ClosureWindow:
    """事件生效后整条链对应的停运窗口。"""

    airport_code: str
    root_event_id: str
    start: datetime
    end: datetime | None  # None == open-ended ("until further notice")
    terminal: bool  # True for a reopened chain

    def contains(self, point: datetime) -> bool:
        # 左闭右开：point == end 时机场已经开放，不算受影响。
        if point < self.start:
            return False
        return self.end is None or point < self.end

    def wait_seconds_from(self, point: datetime) -> int | None:
        """计算指定时刻到窗口末端的真实等待秒数；开放窗口返回 None。

        调用方只在 ``contains(point)`` 为真时调用，因此闭窗口下结果必为
        正秒数（端点相接的 0 秒情况已被左闭右开端点排除）。
        """
        if self.end is None:
            return None
        return wait_seconds(point, self.end)


def chain_window(
    event: DisruptionEvent,
    root: DisruptionEvent,
    airport: Airport,
) -> ClosureWindow:
    """计算应用当前事件后的有效关闭窗口。"""
    if event.event_type == EVENT_CLOSED:
        return ClosureWindow(
            airport_code=event.airport_code,
            root_event_id=root.event_id,
            start=event.effective_from,
            end=event.effective_until,
            terminal=False,
        )
    if event.event_type == EVENT_EXTENDED:
        if event.effective_until is None:  # validated earlier, defensive
            raise ValueError("extended event must define effective_until")
        return ClosureWindow(
            airport_code=event.airport_code,
            root_event_id=root.event_id,
            start=root.effective_from,
            end=event.effective_until,
            terminal=False,
        )
    # reopened: operations resume after the airport's operational buffer
    resume_at = event.effective_from + timedelta(minutes=airport.reopen_buffer_minutes)
    return ClosureWindow(
        airport_code=event.airport_code,
        root_event_id=root.event_id,
        start=root.effective_from,
        end=resume_at,
        terminal=True,
    )


@dataclass(frozen=True)
class EndpointImpact:
    endpoint: str  # "origin" | "destination"
    needed_wait_seconds: int | None  # None == open-ended closure


def _endpoint_impact(
    flight: Flight, endpoint: str, window: ClosureWindow
) -> EndpointImpact | None:
    point = (
        flight.scheduled_departure if endpoint == "origin" else flight.scheduled_arrival
    )
    if not window.contains(point):
        return None
    return EndpointImpact(
        endpoint=endpoint, needed_wait_seconds=window.wait_seconds_from(point)
    )


def classify_flight(
    flight: Flight, window: ClosureWindow
) -> dict[str, Any] | None:
    """返回航班在指定机场的一条影响记录；不受影响时返回 None。"""
    endpoints: list[EndpointImpact] = []
    if flight.origin == window.airport_code:
        impact = _endpoint_impact(flight, "origin", window)
        if impact:
            endpoints.append(impact)
    if flight.destination == window.airport_code:
        impact = _endpoint_impact(flight, "destination", window)
        if impact:
            endpoints.append(impact)
    if not endpoints:
        return None

    if len(endpoints) == 2:
        affected_endpoint = "both"
    else:
        affected_endpoint = endpoints[0].endpoint

    # 业务判定基于精确秒：受最晚才能恢复的端点驱动。任一端点面对开放窗口
    # 则整体无法给出确定时刻。先在秒级取最大值，再统一向上取整到可执行
    # 分钟，保证两端同时受限时结果与端点顺序无关且稳定。
    finite_waits = [e.needed_wait_seconds for e in endpoints if e.needed_wait_seconds is not None]
    if len(finite_waits) < len(endpoints):
        status = IMPACT_PENDING
        wait_secs: int | None = None
        needed_minutes: int | None = None
    else:
        wait_secs = max(finite_waits)  # type: ignore[arg-type]
        needed_minutes = ceil_executable_minutes(wait_secs)
        if flight.can_retime and needed_minutes <= flight.max_delay_minutes:
            status = IMPACT_DELAYED
        else:
            status = IMPACT_CANCELLED

    proposed_departure = proposed_arrival = None
    if status == IMPACT_DELAYED:
        # 建议时刻按最终采用的可执行分钟平移，必然不早于窗口末端（机场实际
        # 开放时刻），即使原始等待只有几十秒。
        proposed_departure = iso_utc(
            flight.scheduled_departure + timedelta(minutes=needed_minutes)  # type: ignore[arg-type]
        )
        proposed_arrival = iso_utc(
            flight.scheduled_arrival + timedelta(minutes=needed_minutes)  # type: ignore[arg-type]
        )

    return {
        "flight_id": flight.flight_id,
        "flight_number": flight.flight_number,
        "airport_code": window.airport_code,
        "affected_endpoint": affected_endpoint,
        "impact_status": status,
        # 原始等待秒数：业务判定的唯一精确依据；开放窗口为 None。
        "wait_seconds": wait_secs,
        # 分钟口径：由精确秒数向上取整得到的最终采用值。
        "overlap_minutes": needed_minutes,
        "delay_minutes": needed_minutes if status == IMPACT_DELAYED else None,
        "proposed_departure": proposed_departure,
        "proposed_arrival": proposed_arrival,
        "passenger_count": flight.passenger_count,
    }


def compute_impacts(
    event: DisruptionEvent,
    root: DisruptionEvent,
    airport: Airport,
    flights: dict[str, Flight],
) -> list[dict[str, Any]]:
    """计算一个事件产生的完整且顺序稳定的影响快照。"""
    window = chain_window(event, root, airport)
    midnight = (
        window.end is not None
        and crosses_local_midnight(window.start, window.end, airport_tz(airport))
    )
    rows: list[dict[str, Any]] = []
    for flight in sorted(flights.values(), key=lambda f: f.flight_id):
        record = classify_flight(flight, window)
        if record is None:
            continue
        record["event_id"] = event.event_id
        record["root_event_id"] = window.root_event_id
        record["crosses_midnight"] = 1 if midnight else 0
        rows.append(record)
    return rows


def airport_tz(airport: Airport):
    from zoneinfo import ZoneInfo

    return ZoneInfo(airport.timezone)
