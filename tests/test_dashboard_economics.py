"""Render tests for the cost and calibration panels.

These assert two things the panels exist to guarantee: that real report data
renders without raising, and that a report predating these analyses produces a
visible placeholder rather than an exception, a blank, or a fabricated zero.
"""

import json

import pytest
from streamlit.testing.v1 import AppTest

from dashboard.components.economics import (
    _calibration_table,
    _cost_bar_frame,
    _reliability_frame,
    render_calibration_panel,
    render_cost_panel,
)


@pytest.fixture(scope="module")
def report():
    with open("ml/artifacts/latest_evaluation_report.json", "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def metrics_payload(report):
    """Shaped exactly like the GET /metrics response the dashboard consumes."""
    return {
        "status": "available",
        "cost_analysis": report.get("cost_analysis"),
        "calibration": report.get("calibration"),
    }


def _run_panel(panel_source: str) -> AppTest:
    # Real report data makes these panels slower than AppTest's 3s default.
    app = AppTest.from_string(panel_source, default_timeout=60)
    app.run(timeout=60)
    return app


def _chart_count(app: AppTest) -> int:
    """st.bar_chart / st.line_chart have no typed AppTest accessor -- they
    surface as UnknownElement. Counting those is the only way to assert a chart
    was actually drawn; `app.get("arrow_bar_chart")` silently returns nothing
    whether or not the chart rendered, which makes it a vacuous assertion."""
    found = []

    def walk(node):
        found.append(type(node).__name__)
        children = getattr(node, "children", None)
        for child in children.values() if isinstance(children, dict) else (children or []):
            walk(child)

    walk(app._tree)
    return found.count("UnknownElement")


def test_cost_bar_frame_has_exactly_the_three_compared_policies(metrics_payload):
    headline = metrics_payload["cost_analysis"]["headline_comparison"]
    frame = _cost_bar_frame(headline)

    assert len(frame) == 3
    assert "Rules-only" in frame.index
    assert "Review everybody" in frame.index
    # Costs must be the real rupee figures, not normalised or rebased values --
    # rebasing to zero would visually exaggerate the model's margin.
    assert frame["Expected cost (₹)"].loc["Rules-only"] == headline["rules_only_cost_inr"]
    assert frame["Expected cost (₹)"].loc["Review everybody"] == headline["review_everybody_cost_inr"]


def test_reliability_frame_includes_the_reference_line_and_drops_empty_bins(metrics_payload):
    frame = _reliability_frame(metrics_payload["calibration"])
    assert frame is not None

    assert "Perfect calibration (y=x)" in frame.columns
    # The reference line must actually be y = x, not a decorative constant.
    assert (frame["Perfect calibration (y=x)"] == frame.index).all()
    # Empty bins are dropped rather than plotted as 0.0, which would invent a
    # data point where the model made no predictions.
    assert frame.index.notna().all()
    assert len(frame) > 0


def test_reliability_frame_draws_logistic_regression_before_and_after(metrics_payload):
    frame = _reliability_frame(metrics_payload["calibration"])
    columns = set(frame.columns)
    assert "Logistic Regression (raw)" in columns
    assert "Logistic Regression (isotonic)" in columns


def test_calibration_table_reports_raw_and_isotonic_per_method(metrics_payload):
    rows = _calibration_table(metrics_payload["calibration"])
    assert rows

    variants_by_method = {}
    for row in rows:
        variants_by_method.setdefault(row["Method"], set()).add(row["Variant"])
        for column in ("Brier", "ECE", "MCE"):
            assert 0.0 <= float(row[column]) <= 1.0

    assert variants_by_method["Logistic Regression"] == {"raw", "isotonic"}


def test_reliability_frame_returns_none_without_curve_data():
    assert _reliability_frame({"methods": {}}) is None
    assert _reliability_frame({}) is None


def test_panels_render_without_exception_on_real_report_data(metrics_payload):
    source = f"""
import json
from dashboard.components.economics import render_cost_panel, render_calibration_panel
payload = json.loads(r'''{json.dumps(metrics_payload)}''')
render_cost_panel(payload)
render_calibration_panel(payload)
"""
    app = _run_panel(source)
    assert not app.exception

    # Both charts must actually be drawn: the cost bars and the reliability curve.
    assert _chart_count(app) == 2

    body = " ".join(
        str(element.value)
        for elements in (app.markdown, app.caption, app.info, app.error)
        for element in elements
    )
    # The margin must be stated in text, not left for the bars to imply.
    assert "margin over reviewing everybody is only" in body
    # The Day 1 cross-link must be carried by the dashboard, not only the model card.
    assert "0.55 threshold in Day 1" in body


def test_panels_render_placeholders_when_sections_are_absent():
    """A report predating these analyses must produce a visible "not present"
    message, never an exception and never a fabricated zero."""
    source = """
from dashboard.components.economics import render_cost_panel, render_calibration_panel
render_cost_panel({"status": "available"})
render_calibration_panel({"status": "available"})
"""
    app = _run_panel(source)
    assert not app.exception

    info_text = " ".join(str(element.value) for element in app.info)
    assert "not present in the current evaluation report" in info_text
    assert "python3 -m ml.evaluate_model" in info_text
    # No chart may be drawn from absent data -- an empty chart would read as
    # "zero cost" rather than "no data".
    assert _chart_count(app) == 0


def test_cost_panel_flags_that_rules_only_loses_to_reviewing_everybody(metrics_payload):
    """The single most counter-intuitive cost finding must be surfaced as an
    explicit callout, not left for a reader to infer from bar heights."""
    headline = metrics_payload["cost_analysis"]["headline_comparison"]
    if headline["rules_only_beats_review_everybody"]:
        pytest.skip("rules-only beats review-everybody in this report; callout not expected")

    source = f"""
import json
from dashboard.components.economics import render_cost_panel
render_cost_panel(json.loads(r'''{json.dumps(metrics_payload)}'''))
"""
    app = _run_panel(source)
    assert not app.exception

    error_text = " ".join(str(element.value) for element in app.error)
    assert "costs more than simply reviewing every merchant-week" in error_text
