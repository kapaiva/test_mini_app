"""Core data loading and calculations"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
from datetime import datetime
import re
import shlex

import pandas as pd

ACRE_TO_FT2 = 43_560.0

SUBBASIN_INPUT_COLUMNS = [
    "ID", "Sub", "area (ac)", "imp (%)", "runoff depth(in)",
    "Peak flow-raw (cfs)", "Total runoff (10^6 gal)",
    "TSS (lb)", "TP (lb)", "TN (lb)", "NO3 (lb)", "PO4 (lb)",
    "Zn (lb)", "Cu (lb)", "Pb (lb)", "As (lb)", "Cr (lb)", "Ni (lb)",
    "Fe (lb)", "Fecal coliform (lb)", "E. coli (lb)",
]

SUPPORTED_PARAMETERS = [
    "Peak Flow", "TSS", "TP", "TN", "NO3", "PO4", "Zn", "Cu", "Pb",
    "As", "Cr", "Ni", "Fe", "Fecal coliform", "E. coli",
]
DECAY_PARAMETERS = SUPPORTED_PARAMETERS[1:]

DATABASE_FILES = [
    "ENR_CCI.csv", "Cost_Database.csv", "Storaged-based-bmp.csv",
    "Infiltration-based-bmp.csv", "decay_rates.csv", "BMP_Efficiencies.csv",
    "BMP_types.csv", "BMP_Cobenefits.csv",
]

COBENEFIT_CATEGORIES = {
    "Air quality improvement": "Environmental",
    "Biodiversity and Ecology": "Environmental",
    "Groundwater Recharge": "Environmental",
    "Temperature reduction": "Environmental",
    "Amenity and Aesthetics": "Social",
    "Food Security": "Social",
    "Recreation and Health": "Social",
    "Pumping and Treatment Reduction": "Economic",
    "Rainwater Harvesting": "Economic",
    "Real Estate Value Appreciation": "Economic",
}


def _clean_name(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).replace("\ufeff", "").replace("\xa0", " ").strip()


def normalize_bmp_dst_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a subbasin table in the canonical user-facing schema."""
    out = df.copy()
    out.columns = [_clean_name(c) for c in out.columns]
    for col in SUBBASIN_INPUT_COLUMNS:
        if col not in out.columns:
            out[col] = "" if col in {"ID", "Sub"} else 0.0
    out = out[SUBBASIN_INPUT_COLUMNS].copy()
    if len(out):
        out["ID"] = range(1, len(out) + 1)
    out["Sub"] = out["Sub"].astype(str).str.strip()
    numeric = [c for c in SUBBASIN_INPUT_COLUMNS if c not in {"ID", "Sub"}]
    out[numeric] = out[numeric].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return out


def load_bmp_types_table(data_dir: str | Path) -> pd.DataFrame:
    df = pd.read_csv(Path(data_dir) / "BMP_types.csv")
    df.columns = [_clean_name(c) for c in df.columns]
    if list(df.columns) != ["BMP Name", "Type"]:
        raise ValueError("BMP_types.csv must contain exactly: BMP Name, Type.")
    df["BMP Name"] = df["BMP Name"].map(_clean_name)
    df["Type"] = df["Type"].map(_clean_name).str.title()
    invalid = df.loc[~df["Type"].isin(["Storage", "Infiltration"]), "BMP Name"]
    if len(invalid):
        raise ValueError("BMP type must be Storage or Infiltration. Check: " + ", ".join(invalid.astype(str)[:10]))
    return df.reset_index(drop=True)


def load_bmp_database_from_csv(data_dir: str | Path) -> pd.DataFrame:
    """Build the normalized BMP table used by costing and solver preprocessing."""
    data_dir = Path(data_dir)
    cost = pd.read_csv(data_dir / "Cost_Database.csv").rename(columns={
        "BMP_Name": "bmp_name",
        "Construction-Cost_usdpercf": "construction_cost_per_ft3",
        "OM_Cost_usdpercf": "om_cost_per_ft3",
        "Default_area_sqf": "area_ft2",
        "Default_depth_ft": "depth_ft",
        "Cost_reference_City": "cost_reference_city",
        "Cost_reference_Year": "cost_reference_year",
    })
    required = [
        "bmp_name", "construction_cost_per_ft3", "om_cost_per_ft3", "area_ft2",
        "depth_ft", "cost_reference_city", "cost_reference_year",
    ]
    missing = [c for c in required if c not in cost.columns]
    if missing:
        raise ValueError("Cost_Database.csv is missing: " + ", ".join(missing))
    cost["bmp_name"] = cost["bmp_name"].map(_clean_name)
    cost["cost_reference_city"] = cost["cost_reference_city"].map(_clean_name)
    for col in required[1:5] + ["cost_reference_year"]:
        cost[col] = pd.to_numeric(cost[col], errors="coerce")
    cost["volume_ft3"] = cost["area_ft2"] * cost["depth_ft"]

    infiltration = pd.read_csv(data_dir / "Infiltration-based-bmp.csv").rename(columns={
        "BMP_Name": "bmp_name",
        "gravel moisture_in": "gravel_moisture",
        "gravel depth_in": "gravel_depth_in",
        "soil moisture": "soil_moisture",
        "soil depth_in": "soil_depth_in",
        "pondind depth_in": "ponding_depth_in",
    })
    infiltration["bmp_name"] = infiltration["bmp_name"].map(_clean_name)

    storage = pd.read_csv(data_dir / "Storaged-based-bmp.csv").rename(
        columns={"BMP_Name": "bmp_name", "v_sqms": "storage_v_sqms"}
    )
    storage["bmp_name"] = storage["bmp_name"].map(_clean_name)

    types = load_bmp_types_table(data_dir)
    type_lookup = dict(zip(types["BMP Name"], types["Type"].str.lower()))
    out = cost.merge(infiltration, on="bmp_name", how="left").merge(storage, on="bmp_name", how="left")
    out["bmp_type"] = out["bmp_name"].map(type_lookup)
    out["co_benefit_score"] = 0.0
    return out


def load_bmp_efficiencies_table(data_dir: str | Path) -> pd.DataFrame:
    """Load intrinsic efficiencies as fractions from 0 to 1."""
    df = pd.read_csv(Path(data_dir) / "BMP_Efficiencies.csv", dtype=object)
    df.columns = [_clean_name(c) for c in df.columns]
    if not len(df.columns) or df.columns[0] != "Parameter":
        raise ValueError("BMP_Efficiencies.csv column A must be Parameter.")
    df["Parameter"] = df["Parameter"].map(_clean_name)

    def as_fraction(value: object) -> float:
        text = _clean_name(value).removesuffix("%").strip()
        try:
            return min(1.0, max(0.0, float(text) / 100.0))
        except ValueError:
            return 0.0

    for col in df.columns[1:]:
        df[col] = df[col].map(as_fraction)
    return df


def target_input_id(parameter: str) -> str:
    return "reduction_" + parameter.lower().replace(" ", "_").replace(".", "").replace("-", "_")


def calculate_target_table(
    subbasins: pd.DataFrame,
    event_duration_seconds: float,
    reduction_percent: dict[str, float],
    target_definitions: list[dict[str, object]],
) -> pd.DataFrame:
    columns = [
        "Group", "Parameter", "Input value", "% reduction", "Target", "Unit",
        "Internal input", "Internal Target", "Internal Unit",
    ]
    if subbasins is None or subbasins.empty:
        return pd.DataFrame(columns=columns)

    duration, rows = float(event_duration_seconds or 0.0), []
    for definition in target_definitions:
        parameter, source = str(definition["parameter"]), str(definition["source_column"])
        before = float(pd.to_numeric(subbasins.get(source, 0.0), errors="coerce").fillna(0.0).sum())
        reduction = min(max(float(reduction_percent.get(parameter, 0.0) or 0.0), 0.0), 100.0)
        target = before * (1.0 - reduction / 100.0)
        is_load = str(definition["method"]) == "load_rate"
        factor = duration if is_load and duration > 0 else 1.0
        rows.append({
            "Group": definition["group"], "Parameter": parameter,
            "Input value": round(before, 6), "% reduction": reduction,
            "Target": round(target, 6), "Unit": str(definition.get("unit", "")),
            "Internal input": round(before / factor, 9),
            "Internal Target": round(target / factor, 9),
            "Internal Unit": str(definition.get("internal_unit", definition.get("unit", ""))),
        })
    return pd.DataFrame(rows, columns=columns)


def _safe_input_id(text: str) -> str:
    value = str(text or "").strip().lower().replace("&", "and")
    return re.sub(r"[^a-z0-9]+", "_", value).strip("_") or "item"


def cobenefit_weight_input_id(criterion: str) -> str:
    return "cb_weight_" + _safe_input_id(criterion)


def cobenefit_score_input_id(criterion: str, bmp_name: str) -> str:
    return "cb_score_" + _safe_input_id(criterion) + "__" + _safe_input_id(bmp_name)


def load_cobenefit_database(data_dir: str | Path) -> pd.DataFrame:
    df = pd.read_csv(Path(data_dir) / "BMP_Cobenefits.csv")
    df.columns = [_clean_name(c) for c in df.columns]
    if list(df.columns[:2]) != ["Co-benefit Name", "Weight Scale"]:
        raise ValueError("BMP_Cobenefits.csv must start with Co-benefit Name, Weight Scale.")
    df["Co-benefit Name"] = df["Co-benefit Name"].map(_clean_name)
    df[df.columns[1:]] = df[df.columns[1:]].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return df


def cobenefit_defaults_from_database(data_dir: str | Path) -> tuple[dict[str, float], dict[str, dict[str, float]], list[str]]:
    df = load_cobenefit_database(data_dir)
    bmps = list(df.columns[2:])
    weights = {str(row["Co-benefit Name"]): float(row["Weight Scale"]) for _, row in df.iterrows()}
    scores = {
        str(row["Co-benefit Name"]): {bmp: float(row[bmp]) for bmp in bmps}
        for _, row in df.iterrows()
    }
    return weights, scores, bmps


def calculate_bmp_cobenefit_scores(
    weights: dict[str, float], scores: dict[str, dict[str, float]]
) -> pd.DataFrame:
    criteria = list(weights)
    bmps = list(dict.fromkeys(bmp for criterion in criteria for bmp in scores.get(criterion, {})))
    total_weight = sum(max(float(weights.get(c, 0.0) or 0.0), 0.0) for c in criteria)
    rows = []
    for bmp in bmps:
        weighted = sum(
            max(float(weights.get(c, 0.0) or 0.0), 0.0)
            * min(max(float(scores.get(c, {}).get(bmp, 0.0) or 0.0), 0.0), 5.0)
            for c in criteria
        )
        rows.append({"BMP": bmp, "Final score": weighted / (total_weight * 5.0) if total_weight else 0.0})
    return pd.DataFrame(rows)

# =============================================================================
# SWMM and user-input helpers
# =============================================================================

# Mapping from SWMM report pollutant names to OptiStorm input columns.
RPT_POLLUTANT_TO_BMPDST: dict[str, str] = {
    "TSS": "TSS (lb)",
    "TP": "TP (lb)",
    "TN": "TN (lb)",
    "NO3": "NO3 (lb)",
    "PO4": "PO4 (lb)",
    "Zn": "Zn (lb)",
    "Cu": "Cu (lb)",
    "Pb": "Pb (lb)",
    "As": "As (lb)",
    "Cr": "Cr (lb)",
    "Ni": "Ni (lb)",
    "Fe": "Fe (lb)",
    "Fecal": "Fecal coliform (lb)",
    "Coli": "E. coli (lb)",
}

# SWMM pollutant display names and unit helpers.
SWMM_POLLUTANT_DISPLAY_NAMES: dict[str, str] = {
    "Fecal": "Fecal coliform",
    "Coli": "E. coli",
}

POLLUTANT_DISPLAY_TO_COLUMN = {
    SWMM_POLLUTANT_DISPLAY_NAMES.get(name, name): column
    for name, column in RPT_POLLUTANT_TO_BMPDST.items()
}

POLLUTANT_DISPLAY_TO_SWMM_NAME: dict[str, str] = {
    SWMM_POLLUTANT_DISPLAY_NAMES.get(name, name): name
    for name in RPT_POLLUTANT_TO_BMPDST
}

BIOLOGICAL_POLLUTANTS = {"Fecal coliform", "E. coli"}
MASS_CONCENTRATION_UNITS = {"MG/L", "UG/L"}
COUNT_CONCENTRATION_UNITS = {"#/L"}

def _pollutant_group(display_name: str) -> str:
    """Return the target group for a detected pollutant."""

    if display_name in {"TP", "TN", "NO3", "PO4"}:
        return "Nutrients"
    if display_name in {"Zn", "Cu", "Pb", "As", "Cr", "Ni", "Fe"}:
        return "Metals"
    if display_name in BIOLOGICAL_POLLUTANTS:
        return "Biological"
    return "Pollutants"

def _internal_unit_for_load_unit(load_unit: str) -> str:
    """Convert a user-facing event-load unit into the internal rate unit."""

    unit = str(load_unit or "load/event")
    if "/event" in unit:
        return unit.replace("/event", "/s")
    return unit + "/s"

def _safe_uploaded_path(uploaded: list[dict[str, Any]] | None) -> str | None:
    """Return the temporary path created by Shiny for an uploaded file.

    Shiny stores uploads as a list of dictionaries. The actual file path is
    stored under ``datapath``. Returning ``None`` makes the downstream reactive
    logic easier to read.
    """

    return uploaded[0]["datapath"] if uploaded else None

def _read_text_lines(path: str | Path) -> list[str]:
    """Read a SWMM text file using common encodings.

    SWMM reports and input files are plain text, but encoding can vary by
    Windows settings. ``latin1`` is a useful fallback because it can read most
    byte sequences without crashing.
    """

    file_path = Path(path)
    for encoding in ("utf-8", "latin1"):
        try:
            return file_path.read_text(encoding=encoding, errors="ignore").splitlines()
        except UnicodeDecodeError:
            continue
    return file_path.read_text(errors="ignore").splitlines()

def _clean_number(value: object, default: float = 0.0) -> float:
    """Convert a text value from SWMM into a float.

    This removes commas and silently returns ``default`` for blank or invalid
    values. That behavior is intentional for input-table creation because
    missing pollutants should become zeros rather than crashing the app.
    """

    try:
        text = str(value).replace(",", "").strip()
        if text == "":
            return default
        return float(text)
    except Exception:
        return default

def _strip_comment(line: str) -> str:
    """Remove SWMM comments marked with a semicolon."""

    return line.split(";", 1)[0].strip()

def _section_from_inp(lines: Iterable[str], section_name: str) -> list[str]:
    """Extract non-empty, non-comment lines from a SWMM .inp section."""

    target = f"[{section_name.upper()}]"
    inside = False
    rows: list[str] = []

    for line in lines:
        stripped = line.strip()

        # SWMM sections look like [OPTIONS], [SUBCATCHMENTS], etc.
        if stripped.startswith("[") and stripped.endswith("]"):
            inside = stripped.upper() == target
            continue

        if inside:
            clean = _strip_comment(line)
            if clean:
                rows.append(clean)

    return rows

def _find_report_heading(lines: list[str], heading: str) -> int | None:
    """Find a SWMM report section heading using whitespace-normalized matching."""

    target = " ".join(heading.lower().split())
    for index, line in enumerate(lines):
        candidate = " ".join(line.strip().lower().split())
        if candidate == target:
            return index
    return None

def _is_numeric_data_row(line: str) -> bool:
    """Return True when a SWMM report line begins with an ID and numeric values."""

    parts = line.split()
    if len(parts) < 2:
        return False

    # First value is the subcatchment ID/name. The second value should be a
    # numeric result. This avoids parsing header rows and dashed separators.
    try:
        float(parts[1].replace(",", ""))
        return True
    except ValueError:
        return False

def _parse_rpt_numeric_table(lines: list[str], heading: str) -> list[list[str]]:
    """Extract numeric rows from a SWMM report summary table.

    SWMM summary tables usually have this structure:

        Heading
        ****************
        blank line
        dashed separator
        header rows
        dashed separator
        numeric rows

    The previous parser stopped at the first asterisk line and therefore never
    reached the numeric rows. This function scans forward until numeric rows are
    found, then stops after the numeric block ends.
    """

    start = _find_report_heading(lines, heading)
    if start is None:
        return []

    rows: list[list[str]] = []
    has_started_data = False

    for line in lines[start + 1 :]:
        stripped = line.strip()

        if _is_numeric_data_row(stripped):
            rows.append(stripped.split())
            has_started_data = True
            continue

        # Once the numeric block has started, a non-numeric line indicates the
        # table has ended. This is safer than stopping at asterisks because the
        # first asterisk line appears before the table body.
        if has_started_data and not stripped:
            break

        if has_started_data and re.search(r"summary\s*$", stripped, flags=re.I):
            break

    return rows

def _parse_rpt_options(lines: list[str]) -> dict[str, str]:
    """Extract key metadata from the SWMM report Analysis Options section."""

    start = _find_report_heading(lines, "Analysis Options")
    if start is None:
        return {}

    options: dict[str, str] = {}
    for line in lines[start + 1 :]:
        # The first large continuity table starts after the options block.
        if line.strip().startswith("**************************"):
            break
        if "..." in line:
            key, value = line.split("...", 1)
            options[key.strip()] = value.strip()

    return options

def _parse_swmm_datetime(date_value: object, time_value: object) -> datetime | None:
    """Parse SWMM date/time strings from .inp or .rpt metadata."""

    date_text = str(date_value or "").strip()
    time_text = str(time_value or "00:00:00").strip() or "00:00:00"
    if not date_text:
        return None

    text = f"{date_text} {time_text}"
    formats = [
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%m-%d-%Y %H:%M:%S",
        "%m-%d-%Y %H:%M",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None

def _swmm_time_to_hours(value: object) -> float | None:
    """Convert a SWMM elapsed-time value to hours.

    Supports decimal hours and ``HH:MM[:SS]`` notation. Returns ``None`` when
    the token is not a time value.
    """

    text = str(value or "").strip()
    if not text:
        return None
    try:
        if ":" not in text:
            return float(text)
        parts = [float(x) for x in text.split(":")]
        if len(parts) == 2:
            return parts[0] + parts[1] / 60.0
        if len(parts) == 3:
            return parts[0] + parts[1] / 60.0 + parts[2] / 3600.0
    except Exception:
        return None
    return None

def rainfall_duration_hours_from_inp(lines: list[str]) -> tuple[float, str]:
    """Estimate design-storm rainfall duration from an uploaded SWMM ``.inp``.

    SWMM does not store a single universal "storm duration" option. When a rain
    gage uses an internal TIMESERIES, however, the duration can be inferred from
    the first and last positive rainfall records plus the gage recording
    interval. If rainfall comes from an external file, the duration cannot be
    determined from the .inp alone and the user must enter it manually.
    """

    gages: list[tuple[str, str, float]] = []
    for line in _section_from_inp(lines, "RAINGAGES"):
        try:
            parts = shlex.split(line)
        except Exception:
            parts = line.split()
        # Name Format Interval SCF Source ...
        if len(parts) < 6:
            continue
        source_type = parts[4].upper()
        interval_hours = _swmm_time_to_hours(parts[2]) or 0.0
        if source_type == "TIMESERIES":
            gages.append((parts[0], parts[5], interval_hours))

    if not gages:
        return 0.0, "No internal rainfall TIMESERIES detected in the .inp file."

    series_rows: dict[str, list[tuple[float, float]]] = {}
    current_date: dict[str, str] = {}

    for line in _section_from_inp(lines, "TIMESERIES"):
        try:
            parts = shlex.split(line)
        except Exception:
            parts = line.split()
        if len(parts) < 3:
            continue

        name = parts[0]
        tokens = parts[1:]
        elapsed_hours: float | None = None
        value: float | None = None

        # If a calendar date was established on a prior row for this series,
        # SWMM allows the date to be omitted from subsequent rows.
        if len(tokens) >= 2 and name in current_date and _swmm_time_to_hours(tokens[0]) is not None:
            dt = _parse_swmm_datetime(current_date[name], tokens[0])
            if dt is not None:
                try:
                    value = float(tokens[1])
                except Exception:
                    value = None
                elapsed_hours = dt.timestamp() / 3600.0
        # Time/value form: Name 0:05 0.12
        elif len(tokens) >= 2 and _swmm_time_to_hours(tokens[0]) is not None:
            elapsed_hours = _swmm_time_to_hours(tokens[0])
            try:
                value = float(tokens[1])
            except Exception:
                value = None
        # Date/time/value form: Name 08/19/2026 12:05 0.12. SWMM allows the
        # date to be omitted on following records, so keep the latest date.
        elif len(tokens) >= 3:
            date_token, time_token = tokens[0], tokens[1]
            dt = _parse_swmm_datetime(date_token, time_token)
            if dt is not None:
                current_date[name] = date_token
                try:
                    value = float(tokens[2])
                except Exception:
                    value = None
                # Store calendar timestamps as Unix-hour-like values; only
                # differences are used later.
                elapsed_hours = dt.timestamp() / 3600.0
        if elapsed_hours is None or value is None:
            continue
        series_rows.setdefault(name, []).append((elapsed_hours, value))

    detected: list[tuple[str, float]] = []
    for gage_name, series_name, interval_hours in gages:
        points = [(t, v) for t, v in series_rows.get(series_name, []) if v > 0]
        if not points:
            continue
        times = [t for t, _ in points]
        duration = max(times) - min(times) + max(interval_hours, 0.0)
        if duration > 0:
            detected.append((gage_name, duration))

    if not detected:
        return 0.0, "Rainfall time series found, but a positive-rainfall duration could not be inferred."

    # Use the longest active rain-gage event so multi-gage models are not
    # truncated when gages have slightly different nonzero periods.
    gage_name, hours = max(detected, key=lambda x: x[1])
    return float(hours), f"Detected from internal rainfall time series ({gage_name})."

def _clean_uploaded_header_name(column_name: object) -> str:
    """Clean a CSV header before comparing it with the required schema.

    The manual CSV template must use the exact user-facing column
    names defined in ``SUBBASIN_INPUT_COLUMNS``. This helper only removes
    accidental byte-order marks, non-breaking spaces, and leading/trailing
    spaces so exported CSV files do not fail for invisible characters.
    It does not rename columns or guess user intent.
    """

    return str(column_name).replace("\ufeff", "").replace("\xa0", " ").strip()

def validate_manual_subbasin_csv(df: pd.DataFrame) -> tuple[bool, str]:
    """Validate that a manually uploaded subbasin CSV matches the template.

    Manual uploads are intentionally strict because this table becomes the
    optimization input. If headers are missing, misspelled, reordered, or if
    unexpected columns are present, the app rejects the file and shows a red
    error message instead of silently filling missing columns with zeros.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw table read from the uploaded CSV file.

    Returns
    -------
    tuple[bool, str]
        ``True`` and an empty message when the CSV headers are valid; otherwise
        ``False`` and a user-facing explanation of what must be corrected.
    """

    uploaded_columns = [_clean_uploaded_header_name(col) for col in df.columns]
    required_columns = SUBBASIN_INPUT_COLUMNS

    missing_columns = [col for col in required_columns if col not in uploaded_columns]
    unexpected_columns = [col for col in uploaded_columns if col not in required_columns]

    if missing_columns or unexpected_columns:
        message_parts: list[str] = [
            "Manual CSV upload failed because the headers do not match the required template."
        ]

        if missing_columns:
            message_parts.append("Missing columns: " + ", ".join(missing_columns))

        if unexpected_columns:
            message_parts.append("Unexpected columns: " + ", ".join(unexpected_columns))

        message_parts.append(
            "Download the subbasin CSV template and keep the column names exactly as provided."
        )

        return False, " ".join(message_parts)

    if uploaded_columns != required_columns:
        return (
            False,
            "Manual CSV upload failed because the headers are present but not in the required order. "
            "Download the subbasin CSV template and keep the same column order.",
        )

    return True, ""

def empty_subbasin_template(n_rows: int = 0) -> pd.DataFrame:
    """Create a blank OptiStorm subbasin input table.

    Parameters
    ----------
    n_rows:
        Number of blank example rows to create. The application uses
        ``n_rows=0`` before any user input is uploaded so that sample data is
        not shown automatically. The download template still uses a few rows
        to make the required CSV format easier to understand.
    """

    if n_rows <= 0:
        return pd.DataFrame(columns=SUBBASIN_INPUT_COLUMNS)

    rows = []
    for i in range(n_rows):
        row = {column: 0.0 for column in SUBBASIN_INPUT_COLUMNS}
        row["ID"] = i + 1
        row["Sub"] = f"Sub{i + 1}"
        rows.append(row)
    return normalize_bmp_dst_columns(pd.DataFrame(rows))

def _parse_inp_pollutant_units(lines: list[str]) -> dict[str, str]:
    """Return pollutant concentration units from the SWMM [POLLUTANTS] section.

    The pollutant name alone does not determine meaning. For example, Fecal and
    Coli defined as MG/L are treated as mass/surrogate pollutants, while #/L is
    count-based. This dictionary is used only for clear Step 1 diagnostics.
    """

    units: dict[str, str] = {}
    for line in _section_from_inp(lines, "POLLUTANTS"):
        clean = _strip_comment(line)
        if not clean or clean.startswith(";"):
            continue
        parts = clean.split()
        if len(parts) < 2:
            continue
        name = parts[0].strip()
        unit = parts[1].strip().upper()
        if name and not name.startswith(";"):
            units[name] = unit
    return units

def parse_swmm_inp(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Parse subbasin area, imperviousness, and model time steps from SWMM .inp."""

    lines = _read_text_lines(path)
    rainfall_duration_hours, rainfall_duration_note = rainfall_duration_hours_from_inp(lines)

    subcatchment_rows: list[dict[str, Any]] = []
    for line in _section_from_inp(lines, "SUBCATCHMENTS"):
        parts = line.split()

        # SWMM [SUBCATCHMENTS] columns:
        # Name, RainGage, Outlet, Area, %Imperv, Width, %Slope, CurbLen, SnowPack
        if len(parts) < 5:
            continue

        subcatchment_rows.append(
            {
                "Sub": parts[0],
                "area (ac)": _clean_number(parts[3]),
                "imp (%)": _clean_number(parts[4]),
            }
        )

    subcatchments = pd.DataFrame(subcatchment_rows)
    if not subcatchments.empty:
        subcatchments = subcatchments.drop_duplicates("Sub", keep="first")

    options: dict[str, str] = {}
    for line in _section_from_inp(lines, "OPTIONS"):
        parts = line.split(None, 1)
        if len(parts) == 2:
            options[parts[0].upper()] = parts[1].strip()

    pollutant_units = _parse_inp_pollutant_units(lines)

    metadata = {
        "source_inp": "uploaded" if not subcatchments.empty else "not detected",
        "n_subcatchments_in_inp": int(len(subcatchments)),
        "flow_units": options.get("FLOW_UNITS", ""),
        "infiltration": options.get("INFILTRATION", ""),
        "flow_routing": options.get("FLOW_ROUTING", ""),
        "report_step": options.get("REPORT_STEP", ""),
        "routing_step": options.get("ROUTING_STEP", ""),
        "wet_step": options.get("WET_STEP", ""),
        "dry_step": options.get("DRY_STEP", ""),
        "start_date": options.get("START_DATE", ""),
        "start_time": options.get("START_TIME", ""),
        "end_date": options.get("END_DATE", ""),
        "end_time": options.get("END_TIME", ""),
        "rainfall_duration_hours_detected": rainfall_duration_hours,
        "rainfall_duration_source": rainfall_duration_note,
        "pollutant_units_detected": ", ".join(f"{k}: {v}" for k, v in pollutant_units.items()),
        "_pollutant_units": pollutant_units,
    }

    return subcatchments, metadata

def parse_swmm_rpt(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Parse runoff, peak flow, and pollutant loads from a SWMM .rpt file."""

    lines = _read_text_lines(path)

    runoff_rows: list[dict[str, Any]] = []
    for parts in _parse_rpt_numeric_table(lines, "Subcatchment Runoff Summary"):
        if len(parts) < 10:
            continue

        # SWMM runoff summary numeric columns:
        # Sub, Precip, Runon, Evap, Infil, ImpervRunoff, PervRunoff,
        # TotalRunoffDepth, TotalRunoffMG, PeakRunoffCFS, RunoffCoeff
        runoff_depth_in = _clean_number(parts[7])
        total_runoff_mg = _clean_number(parts[8])

        # Fallback area when no .inp is uploaded.
        # 1 inch over 1 acre = 27,154 gallons.
        area_ac = 0.0
        if runoff_depth_in > 0:
            area_ac = total_runoff_mg * 1_000_000.0 / (runoff_depth_in * 27_154.0)

        runoff_rows.append(
            {
                "Sub": parts[0],
                "area (ac)": area_ac,
                "runoff depth(in)": runoff_depth_in,
                "Peak flow-raw (cfs)": _clean_number(parts[9]),
                "Total runoff (10^6 gal)": total_runoff_mg,
            }
        )

    runoff = pd.DataFrame(runoff_rows)

    # Parse pollutant names from the Subcatchment Washoff Summary header when
    # possible. This avoids assuming a fixed order when future SWMM reports add
    # TP, TN, metals, etc.
    pollutant_names = _detect_washoff_pollutant_names(lines)
    if not pollutant_names:
        pollutant_names = ["TSS", "Fecal", "Coli"]

    washoff_rows: list[dict[str, Any]] = []
    for parts in _parse_rpt_numeric_table(lines, "Subcatchment Washoff Summary"):
        if len(parts) < 2:
            continue

        row: dict[str, Any] = {"Sub": parts[0]}
        for pollutant, value in zip(pollutant_names, parts[1:]):
            target_column = RPT_POLLUTANT_TO_BMPDST.get(pollutant)
            if target_column:
                row[target_column] = _clean_number(value)
        washoff_rows.append(row)

    washoff = pd.DataFrame(washoff_rows)

    if runoff.empty and washoff.empty:
        # Do not create sample subbasins when a report cannot be parsed.
        # This makes parsing failures visible instead of silently showing
        # placeholder data.
        subbasin_inputs = empty_subbasin_template(0)
    elif runoff.empty:
        subbasin_inputs = washoff.copy()
    elif washoff.empty:
        subbasin_inputs = runoff.copy()
    else:
        subbasin_inputs = runoff.merge(washoff, on="Sub", how="left")

    subbasin_inputs = normalize_bmp_dst_columns(subbasin_inputs)
    options = _parse_rpt_options(lines)

    metadata = {
        "source_rpt": "uploaded",
        "n_subcatchments_in_rpt": int(len(subbasin_inputs)),
        "flow_units": options.get("Flow Units", ""),
        "report_step_rpt": options.get("Report Time Step", ""),
        "routing_step_rpt": options.get("Routing Time Step", ""),
        "wet_step_rpt": options.get("Wet Time Step", ""),
        "dry_step_rpt": options.get("Dry Time Step", ""),
        "start_date": options.get("Start Date", options.get("Starting Date", "")),
        "start_time": options.get("Start Time", options.get("Starting Time", "")),
        "end_date": options.get("End Date", options.get("Ending Date", "")),
        "end_time": options.get("End Time", options.get("Ending Time", "")),
    }

    return subbasin_inputs, metadata

def _detect_washoff_pollutant_names(lines: list[str]) -> list[str]:
    """Detect pollutant columns listed in the SWMM washoff summary header."""

    start = _find_report_heading(lines, "Subcatchment Washoff Summary")
    if start is None:
        return []

    known_names = set(RPT_POLLUTANT_TO_BMPDST)
    for line in lines[start + 1 : start + 15]:
        tokens = line.split()
        found = [token for token in tokens if token in known_names]
        if found:
            return found

    return []

def combine_rpt_and_inp(
    rpt_df: pd.DataFrame | None,
    inp_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Merge .rpt results with .inp area and imperviousness.

    The .rpt file is treated as the source of hydrologic/water-quality results.
    The .inp file is treated as the source of physical subbasin properties. When
    both are uploaded, .inp area and imperviousness overwrite .rpt-derived area.
    """

    rpt_df = pd.DataFrame() if rpt_df is None else rpt_df.copy()
    inp_df = pd.DataFrame() if inp_df is None else inp_df.copy()

    if rpt_df.empty and inp_df.empty:
        # Before the user uploads SWMM files or a manual CSV, keep the table
        # empty. This prevents the Step 1 page from showing default/sample
        # subbasins automatically.
        return empty_subbasin_template(0)

    if rpt_df.empty:
        base = inp_df.copy()
    else:
        base = rpt_df.copy()

    if not inp_df.empty:
        inp_small = inp_df[["Sub", "area (ac)", "imp (%)"]].drop_duplicates("Sub")
        base = base.drop(columns=[c for c in ["area (ac)", "imp (%)"] if c in base.columns])
        base = base.merge(inp_small, on="Sub", how="left")

    out = normalize_bmp_dst_columns(base)
    if not out.empty:
        # Use clean sequential names for display and downstream optimization.
        # The merge with .inp/.rpt is already complete at this point.
        out["Sub"] = [f"Sub{i}" for i in range(1, len(out) + 1)]
    return out

def _display_pollutant_name(swmm_name: str) -> str:
    """Return a user-friendly pollutant label for SWMM report names."""

    return SWMM_POLLUTANT_DISPLAY_NAMES.get(str(swmm_name), str(swmm_name))

def _swmm_name_from_display(display_name: str) -> str:
    """Return the SWMM pollutant name for a user-facing pollutant label."""

    return POLLUTANT_DISPLAY_TO_SWMM_NAME.get(str(display_name), str(display_name))

def _present_pollutants(subbasins: pd.DataFrame) -> list[str]:
    """Return pollutant display names that have nonzero values in Step 1 data."""

    if subbasins is None or subbasins.empty:
        return []

    present: list[str] = []
    for swmm_name, column in RPT_POLLUTANT_TO_BMPDST.items():
        if column not in subbasins.columns:
            continue
        values = pd.to_numeric(subbasins[column], errors="coerce").fillna(0.0)
        if float(values.sum()) > 0:
            present.append(_display_pollutant_name(swmm_name))
    return present

def _event_load_unit(display_name: str, swmm_unit: str) -> str:
    """Return the user-facing event-load unit."""

    unit = str(swmm_unit or "").upper().strip()
    if unit in COUNT_CONCENTRATION_UNITS:
        return "counts/event"
    if unit in MASS_CONCENTRATION_UNITS and display_name in BIOLOGICAL_POLLUTANTS:
        return "surrogate load/event"
    if unit in MASS_CONCENTRATION_UNITS:
        # SWMM mass loads are stored in the current input schema as lb/event.
        return "lb/event"
    return "model units/event"

def active_target_definitions(
    subbasins: pd.DataFrame,
    inp_metadata: dict[str, Any],
    rpt_metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return only targets that are present in the imported Step 1 data."""

    if subbasins is None or subbasins.empty:
        return []

    definitions: list[dict[str, Any]] = []
    peak = pd.to_numeric(subbasins.get("Peak flow-raw (cfs)", 0.0), errors="coerce").fillna(0.0)
    if float(peak.sum()) > 0:
        definitions.append({
            "group": "Physical",
            "parameter": "Peak Flow",
            "source_column": "Peak flow-raw (cfs)",
            "method": "sum",
            "unit": "cfs",
            "internal_unit": "cfs",
        })

    unit_lookup = dict(inp_metadata.get("_pollutant_units", {}) or {})
    for display in _present_pollutants(subbasins):
        column = POLLUTANT_DISPLAY_TO_COLUMN.get(display, "")
        if not column:
            continue
        swmm_unit = unit_lookup.get(_swmm_name_from_display(display), "")
        load_unit = _event_load_unit(display, swmm_unit)
        definitions.append({
            "group": _pollutant_group(display),
            "parameter": display,
            "source_column": column,
            "method": "load_rate",
            "unit": load_unit,
            "internal_unit": _internal_unit_for_load_unit(load_unit),
        })

    return definitions

def display_subbasins_for_preview(df: pd.DataFrame) -> pd.DataFrame:
    """Hide absent all-zero pollutant columns in the Step 1 preview only."""

    if df is None or df.empty:
        return df
    out = df.copy()
    for column in list(RPT_POLLUTANT_TO_BMPDST.values()):
        if column in out.columns:
            total = pd.to_numeric(out[column], errors="coerce").fillna(0.0).sum()
            if float(total) == 0.0:
                out = out.drop(columns=[column])
    return out
