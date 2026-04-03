"""DB↔Model変換ユーティリティ。"""

import sqlite3
from typing import TypeVar

T = TypeVar("T")


def from_row(row: sqlite3.Row, model_cls: type[T]) -> T:
    """sqlite3.Row を dataclass に変換する。"""
    return model_cls(**{f: row[f] for f in model_cls.__dataclass_fields__})


def columns_of(model_cls: type) -> str:
    """dataclass のフィールド名をSQLカラム列挙文字列として返す。"""
    return ", ".join(model_cls.__dataclass_fields__)
