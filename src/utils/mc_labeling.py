from __future__ import annotations

import argparse
import logging

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunStats:
    csv_found: int = 0
    csv_updated: int = 0
    csv_skipped: int = 0
    csv_failed: int = 0
    txt_found: int = 0
    txt_updated: int = 0
    txt_skipped: int = 0
    txt_failed: int = 0


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add multiclass labels to *_labeled.csv files under a dataset directory, "
            "and append human-readable timestamps to .txt labeling helper files."
        )
    )
    parser.add_argument(
        "--dataset_path",
        required=True,
        type=str,
        help="Directory containing labeled CSVs and txt files (searched recursively).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Do not modify files; only log what would change.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def _validate_dataset_dir(dataset_dir: Path) -> None:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"dataset_path does not exist: {dataset_dir}")
    if not dataset_dir.is_dir():
        raise ValueError(f"dataset_path must be a directory: {dataset_dir}")


def _find_labeled_csvs(dataset_dir: Path) -> list[Path]:
    return sorted(p for p in dataset_dir.rglob("*_labeled.csv") if p.is_file())


def _find_txt_files(dataset_dir: Path) -> list[Path]:
    return sorted(p for p in dataset_dir.rglob("*.txt") if p.is_file())


def _resolve_bin_label_column(df: pd.DataFrame) -> Optional[str]:
    """
    Prefer Bin_Label, but tolerate BinLabel for backward compatibility.
    """
    if "Bin_Label" in df.columns:
        return "Bin_Label"
    if "BinLabel" in df.columns:
        return "BinLabel"
    return None


def _update_csv_mc_label(csv_path: Path, dry_run: bool) -> bool:
    """
    Returns True if the file would be/was updated; False if skipped.
    """
    logger.debug("Processing CSV: %s", csv_path)
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        logger.exception("Failed reading CSV: %s", csv_path)
        raise

    bin_col = _resolve_bin_label_column(df)
    if bin_col is None:
        logger.warning(
            "Skipping CSV (no Bin_Label/BinLabel column): %s", csv_path)
        return False

    desired = df[bin_col].map({0: "Benign", 1: "FILL_ME"})

    unknown_mask = desired.isna()
    if unknown_mask.any():
        bad_vals = sorted(set(df.loc[unknown_mask, bin_col].dropna().tolist()))
        logger.warning(
            "CSV has unexpected %s values %s; mapping them to FILL_ME: %s",
            bin_col,
            bad_vals,
            csv_path,
        )
        desired = desired.fillna("FILL_ME")

    before = df["MC_Label"] if "MC_Label" in df.columns else None
    df["MC_Label"] = desired

    changed = True
    if before is not None:
        try:
            changed = not before.astype(str).equals(df["MC_Label"].astype(str))
        except Exception:
            changed = True

    if not changed:
        logger.info("No change needed (MC_Label already matches): %s", csv_path)
        return False

    if dry_run:
        logger.info("[dry-run] Would update MC_Label in: %s", csv_path)
        return True

    tmp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
    try:
        df.to_csv(tmp_path, index=False)
        tmp_path.replace(csv_path)
        logger.info("Updated MC_Label in: %s", csv_path)
        return True
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                logger.debug("Could not remove temp file: %s", tmp_path)


def _extract_unix_time_from_third_field(third_field: str) -> Optional[float]:
    """
    Expects something like: "Unix Time: 1751471126.992594"
    """
    if "Unix Time:" not in third_field:
        return None
    _, _, tail = third_field.partition("Unix Time:")
    tail = tail.strip()
    if not tail:
        return None
    try:
        return float(tail)
    except ValueError:
        return None


def _format_unix_time(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _update_txt_file(txt_path: Path, dry_run: bool) -> bool:
    """
    Appends ', <human datetime>' to lines that have a parsable unix time in the
    third CSV field and do not already appear to have an appended datetime.

    Returns True if file would be/was updated, else False.
    """
    logger.debug("Processing TXT: %s", txt_path)

    try:
        original_lines = txt_path.read_text(
            encoding="utf-8", errors="replace").splitlines(True)
    except Exception:
        logger.exception("Failed reading TXT: %s", txt_path)
        raise

    updated_any = False
    new_lines: list[str] = []

    for line in original_lines:
        stripped = line.rstrip("\n")
        parts = [p.strip() for p in stripped.split(",")]

        if len(parts) < 3:
            new_lines.append(line)
            continue

        unix_ts = _extract_unix_time_from_third_field(parts[2])
        if unix_ts is None:
            new_lines.append(line)
            continue

        if len(parts) >= 4:
            if len(parts[3]) >= 19 and parts[3][4] == "-" and parts[3][7] == "-" and ":" in parts[3]:
                new_lines.append(line)
                continue

        human = _format_unix_time(unix_ts)
        new_line = f"{stripped}, {human}\n"
        new_lines.append(new_line)
        updated_any = True

    if not updated_any:
        logger.info("No TXT changes needed: %s", txt_path)
        return False

    if dry_run:
        logger.info("[dry-run] Would append datetimes in: %s", txt_path)
        return True

    tmp_path = txt_path.with_suffix(txt_path.suffix + ".tmp")
    try:
        tmp_path.write_text("".join(new_lines), encoding="utf-8")
        tmp_path.replace(txt_path)
        logger.info("Updated TXT (appended datetimes): %s", txt_path)
        return True
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                logger.debug("Could not remove temp file: %s", tmp_path)


def main() -> None:
    args = _parse_arguments()
    _configure_logging(args.verbose)

    dataset_dir = Path(args.dataset_path).expanduser().resolve()

    try:
        _validate_dataset_dir(dataset_dir)
    except Exception as exc:
        logger.error("Invalid dataset_path: %s", exc)
        raise SystemExit(1) from exc

    stats = RunStats()

    csv_files = _find_labeled_csvs(dataset_dir)
    stats = RunStats(
        csv_found=len(csv_files),
        txt_found=stats.txt_found,
        csv_updated=stats.csv_updated,
        csv_skipped=stats.csv_skipped,
        csv_failed=stats.csv_failed,
        txt_updated=stats.txt_updated,
        txt_skipped=stats.txt_skipped,
        txt_failed=stats.txt_failed,
    )
    logger.info("Found %d labeled CSVs under %s", len(csv_files), dataset_dir)

    csv_updated = csv_skipped = csv_failed = 0
    for csv_path in csv_files:
        try:
            did_update = _update_csv_mc_label(csv_path, dry_run=args.dry_run)
            if did_update:
                csv_updated += 1
            else:
                csv_skipped += 1
        except Exception:
            csv_failed += 1
            logger.exception("CSV failed: %s", csv_path)

    txt_files = _find_txt_files(dataset_dir)
    logger.info("Found %d txt files under %s", len(txt_files), dataset_dir)

    txt_updated = txt_skipped = txt_failed = 0
    for txt_path in txt_files:
        try:
            did_update = _update_txt_file(txt_path, dry_run=args.dry_run)
            if did_update:
                txt_updated += 1
            else:
                txt_skipped += 1
        except Exception:
            txt_failed += 1
            logger.exception("TXT failed: %s", txt_path)

    logger.info(
        "Done. CSVs: found=%d updated=%d skipped=%d failed=%d | "
        "TXTs: found=%d updated=%d skipped=%d failed=%d",
        len(csv_files),
        csv_updated,
        csv_skipped,
        csv_failed,
        len(txt_files),
        txt_updated,
        txt_skipped,
        txt_failed,
    )


if __name__ == "__main__":
    main()
