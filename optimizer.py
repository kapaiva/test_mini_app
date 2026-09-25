"""Optimization model"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pulp

from data_processing import SUPPORTED_PARAMETERS

TARGET_COLUMNS = SUPPORTED_PARAMETERS

@dataclass
class OptimizationConfig:
    objective: str = "Cost Only"
    lambda_cobenefit: float = 0.0
    cobenefit_scores: dict[str, float] | None = None
    subbasin_taf_caps: dict[str, float] | None = None
    solver_time_limit_sec: int = 120


def _target_lookup(targets: pd.DataFrame) -> dict[str, float]:
    if targets is None or targets.empty:
        return {}
    out = {}
    for _, row in targets.iterrows():
        parameter = str(row.get("Parameter", "")).strip()
        value = pd.to_numeric(row.get("Internal Target"), errors="coerce")
        if parameter and pd.notna(value):
            out[parameter] = float(value)
    return out


def solve_bmp_placement_from_coefficients(
    coefficients: pd.DataFrame,
    targets: pd.DataFrame,
    config: OptimizationConfig | None = None,
) -> dict:
    """Solve continuous BMP allocation fractions for each subbasin."""
    config = config or OptimizationConfig()
    if coefficients is None or coefficients.empty:
        return {
            "status": "Missing coefficient matrix",
            "objective_value": None,
            "placement": pd.DataFrame(),
            "performance": pd.DataFrame(),
            "active_constraints": pd.DataFrame(),
        }

    coef = coefficients.copy()
    coef["Subbasin"] = coef["Subbasin"].astype(str)
    coef["BMP"] = coef["BMP"].astype(str)
    coef["Cost coefficient"] = pd.to_numeric(coef["Cost coefficient"], errors="coerce").fillna(0.0)

    model = pulp.LpProblem("OptiStorm_BMP_Placement", pulp.LpMinimize)
    x: dict[tuple[str, str], pulp.LpVariable] = {}
    for _, row in coef.iterrows():
        sub, bmp = row["Subbasin"], row["BMP"]
        ub = min(1.0, max(0.0, float(row.get("Decision upper bound", 1.0) or 0.0)))
        name = f"x__{sub}__{bmp}".replace(" ", "_").replace(".", "")
        x[(sub, bmp)] = pulp.LpVariable(name, lowBound=0.0, upBound=ub, cat=pulp.LpContinuous)

    for sub, group in coef.groupby("Subbasin"):
        model += pulp.lpSum(x[(sub, str(row["BMP"]))] for _, row in group.iterrows()) == 1.0, f"allocation_{sub}"

    cap_lookup = config.subbasin_taf_caps or {}
    cap_rows = []
    for sub, group in coef.groupby("Subbasin"):
        cap = min(1.0, max(0.0, float(cap_lookup.get(str(sub), 1.0))))
        real_bmps = [
            (str(sub), str(row["BMP"]))
            for _, row in group.iterrows()
            if str(row["BMP"]) != "No BMP" and float(row.get("Decision upper bound", 1.0) or 0.0) > 1e-12
        ]
        if real_bmps:
            model += pulp.lpSum(x[key] for key in real_bmps) <= cap, f"taf_cap_{str(sub).replace(' ', '_').replace('.', '')}"
        cap_rows.append({"Subbasin": str(sub), "TAF cap": cap, "Available real BMP variables": len(real_bmps)})

    lambda_cb = min(1.0, max(0.0, float(config.lambda_cobenefit or 0.0)))
    scores = config.cobenefit_scores or {}
    use_cobenefits = str(config.objective).strip().casefold() == "cost + co-benefits"

    cb_scores, multipliers, adjusted_costs, objective_terms = [], [], [], []
    for _, row in coef.iterrows():
        sub, bmp = row["Subbasin"], row["BMP"]
        score = 0.0 if bmp == "No BMP" else min(1.0, max(0.0, float(scores.get(bmp, 0.0) or 0.0)))
        multiplier = 1.0 - lambda_cb * score if use_cobenefits else 1.0
        adjusted = float(row["Cost coefficient"]) * multiplier
        cb_scores.append(score)
        multipliers.append(multiplier)
        adjusted_costs.append(adjusted)
        objective_terms.append(adjusted * x[(sub, bmp)])

    coef["Co-benefit score"] = cb_scores
    coef["Objective multiplier"] = multipliers
    coef["Adjusted cost coefficient"] = adjusted_costs
    model += pulp.lpSum(objective_terms)

    target_values = _target_lookup(targets)
    active_constraints = []
    no_bmp = coef[coef["BMP"] == "No BMP"].drop_duplicates("Subbasin")
    for parameter in TARGET_COLUMNS:
        if parameter not in coef.columns or parameter not in target_values:
            continue
        coef[parameter] = pd.to_numeric(coef[parameter], errors="coerce").fillna(0.0)
        baseline = float(no_bmp[parameter].sum()) if parameter in no_bmp.columns else 0.0
        target = float(target_values[parameter])
        if baseline <= 0 or target < 0 or target >= baseline:
            continue
        expr = pulp.lpSum(float(row[parameter]) * x[(row["Subbasin"], row["BMP"])] for _, row in coef.iterrows())
        model += expr <= target, f"target_{parameter.replace(' ', '_').replace('.', '')}"
        active_constraints.append({
            "Parameter": parameter,
            "Baseline": baseline,
            "Target": target,
            "Required reduction": baseline - target,
        })

    status_code = model.solve(pulp.COIN_CMD(msg=False, timeLimit=config.solver_time_limit_sec))
    status = pulp.LpStatus.get(status_code, "Unknown")

    placement_rows = []
    for _, row in coef.iterrows():
        sub, bmp = row["Subbasin"], row["BMP"]
        decision = float(pulp.value(x[(sub, bmp)]) or 0.0)
        if decision <= 1e-7:
            continue
        max_units = float(row.get("Max BMP units across active targets", 0.0) or 0.0)
        raw_cost = float(row.get("Cost coefficient", 0.0) or 0.0)
        adjusted_cost = float(row.get("Adjusted cost coefficient", raw_cost) or 0.0)
        placement_rows.append({
            "subbasin": sub,
            "bmp_name": bmp,
            "decision_value": decision,
            "max_units_across_targets": max_units,
            "estimated_units": decision * max_units,
            "cost_coefficient": raw_cost,
            "total_cost": decision * raw_cost,
            "cobenefit_score": float(row.get("Co-benefit score", 0.0) or 0.0),
            "objective_multiplier": float(row.get("Objective multiplier", 1.0) or 0.0),
            "adjusted_cost_coefficient": adjusted_cost,
            "objective_contribution": decision * adjusted_cost,
            "decision_upper_bound": float(row.get("Decision upper bound", 1.0) or 0.0),
            "availability_note": str(row.get("Exclusion reason", "") or ""),
        })
    placement = pd.DataFrame(placement_rows)

    performance_rows = []
    for parameter in TARGET_COLUMNS:
        if parameter not in coef.columns:
            continue
        before = float(no_bmp[parameter].sum()) if parameter in no_bmp.columns else 0.0
        after = sum(
            float(pulp.value(x[(row["Subbasin"], row["BMP"])]) or 0.0) * float(row.get(parameter, 0.0) or 0.0)
            for _, row in coef.iterrows()
        )
        performance_rows.append({"Parameter": parameter, "Before": before, "After": after, "Reduction": before - after})
    performance = pd.DataFrame(performance_rows)

    cap_audit = pd.DataFrame(cap_rows)
    if not cap_audit.empty:
        achieved = []
        for sub in cap_audit["Subbasin"].astype(str):
            group = coef[(coef["Subbasin"] == sub) & (coef["BMP"] != "No BMP")]
            total = sum(
                float(pulp.value(x[(sub, str(row["BMP"]))]) or 0.0)
                for _, row in group.iterrows()
                if float(row.get("Decision upper bound", 1.0) or 0.0) > 1e-12
            )
            achieved.append(total)
        cap_audit["Real BMP TAF sum"] = achieved
        cap_audit["Remaining No BMP fraction"] = (1.0 - cap_audit["Real BMP TAF sum"]).clip(0.0, 1.0)
        cap_audit["Cap binding"] = (cap_audit["TAF cap"] - cap_audit["Real BMP TAF sum"]).abs() <= 1e-6

    return {
        "status": status,
        "objective_value": float(pulp.value(model.objective) or 0.0),
        "objective_name": config.objective,
        "lambda_cobenefit": lambda_cb if use_cobenefits else 0.0,
        "placement": placement,
        "performance": performance,
        "active_constraints": pd.DataFrame(active_constraints),
        "allocation_caps": cap_audit,
        "coefficient_matrix": coef,
    }
