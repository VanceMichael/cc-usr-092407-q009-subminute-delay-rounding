"""时间处理工具。

事件输入可以携带不同 UTC 偏移，区间比较前统一转换为带时区的 UTC 时间。
机场本地日期只用于判断是否跨午夜。
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.errors import ValidationError

# Accept ISO 8601 date-time with Z, +HH:MM or explicit timezone.
# Naive timestamps are rejected: a disruption window without a zone is ambiguous.
_OFFSET_RE = re.compile(r"(Z|[+-]\d{2}:\d{2})$")
# The domain operates at whole-second precision. Reject fractional seconds so
# that storage (whole-second ISO strings) and classification can never diverge
# due to serialization truncation after a restart.
_FRACTION_RE = re.compile(r"T\d{2}:\d{2}:\d{2}[.,]\d")


def parse_event_datetime(value: object, field: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValidationError(
            f"Field '{field}' must be an ISO 8601 date-time string",
            {"field": field},
        )
    text = value.strip()
    if not _OFFSET_RE.search(text):
        raise ValidationError(
            f"Field '{field}' must include a timezone designator (Z or ±HH:MM)",
            {"field": field, "received": value},
        )
    if _FRACTION_RE.search(text):
        raise ValidationError(
            f"Field '{field}' must use whole-second precision (no fractional seconds)",
            {"field": field, "received": value},
        )
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError:
        raise ValidationError(
            f"Field '{field}' is not a valid ISO 8601 date-time",
            {"field": field, "received": value},
        ) from None
    if dt.tzinfo is None:  # pragma: no cover - guarded by regex above
        raise ValidationError(
            f"Field '{field}' must include a timezone designator (Z or ±HH:MM)",
            {"field": field, "received": value},
        )
    return dt.astimezone(timezone.utc)


def to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        raise ValueError("datetime must be timezone-aware")
    return dt.astimezone(timezone.utc)


def load_timezone(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        raise ValidationError(
            f"Airport timezone '{name}' is not a valid IANA timezone",
            {"timezone": name},
        ) from None


def overlaps(start_a: datetime, end_a: datetime, start_b: datetime, end_b: datetime) -> bool:
    """计算左闭右开区间的重叠，端点相接不算重叠。"""
    return start_a < end_b and start_b < end_a


def wait_seconds(point: datetime, end: datetime) -> int:
    """从 ``point`` 等到窗口末端 ``end`` 的整秒数。

    比较基于时区感知 datetime（内部统一为 UTC），因此夏令时切换不会产生
    歧义。结果只表示真实经过的秒数，不做任何分钟展示舍入——业务判定必须
    基于这个精确值，而不能从分钟口径反推。
    """
    delta = (end - point).total_seconds()
    # datetime 相减对感知时间给出精确秒（含微秒）；输入按整秒解析，这里
    # 四舍五入到最近整秒以吸收任何亚秒级误差。
    return int(round(delta))


def ceil_executable_minutes(seconds: int) -> int:
    """把任意非负等待秒数向上取到可执行分钟。

    0 秒（时刻恰好落在左闭右开窗口的末端，本就不受影响）取 0；任何正的
    等待，哪怕只有 1 秒，都必须取到下一个整分钟，否则建议时刻会早于机场
    实际开放时刻。
    """
    if seconds <= 0:
        return 0
    return -(-seconds // 60)


def overlap_minutes(
    start_a: datetime, end_a: datetime, start_b: datetime, end_b: datetime
) -> int:
    """左闭右开区间交集长度，按可执行分钟向上取整。"""
    start = max(start_a, start_b)
    end = min(end_a, end_b)
    return ceil_executable_minutes(wait_seconds(start, end))


def crosses_local_midnight(
    start: datetime, end: datetime, tz: ZoneInfo
) -> bool:
    """判断左闭右开区间在指定时区内是否跨越两个自然日。

    比较的是机场本地日历日，UTC 偏移（含夏令时跳变）由 zoneinfo 负责，
    因此春季跳表/秋季回表都能得到稳定结果。
    """
    local_start = start.astimezone(tz)
    local_end = end.astimezone(tz)
    return local_start.date() != local_end.date()


def minutes_until(start: datetime, end: datetime) -> int:
    """从 start 到 end 的可执行分钟（正等待向上取整）。"""
    return ceil_executable_minutes(wait_seconds(start, end))
