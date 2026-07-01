"""Endpoint tests for the read-only dashboard API.

Cover each endpoint against seeded core data, the verbatim-serialization
guarantee, the honest empty/day-one states, and — most importantly — that NO
mutating endpoint exists anywhere on the app.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import create_app
from api.serializers import EQUITY_CURVE_NOTICE

SURVIVOR_HASH = "is-survivor-hash"


# ---------------------------------------------------------------------------
# The single most important guarantee: read-only
# ---------------------------------------------------------------------------


def test_no_mutating_endpoint_exists() -> None:
    app = create_app()
    mutating = {"POST", "PUT", "PATCH", "DELETE"}
    offenders = []
    for route in app.routes:
        methods = getattr(route, "methods", set()) or set()
        if methods & mutating:
            offenders.append((getattr(route, "path", "?"), sorted(methods & mutating)))
    assert not offenders, f"mutating routes exist: {offenders}"


def test_cors_allows_only_get() -> None:
    # The CORS middleware is configured allow_methods=["GET"]; assert the app
    # never advertises a write method on a preflight.
    app = create_app()
    for route in app.routes:
        methods = getattr(route, "methods", set()) or set()
        assert methods <= {"GET", "HEAD", "OPTIONS"}, f"{getattr(route, 'path', '?')}: {methods}"


# ---------------------------------------------------------------------------
# health
# ---------------------------------------------------------------------------


def test_health(seeded_client: TestClient) -> None:
    r = seeded_client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["mode"] == "read-only"
    assert body["paper_only"] is True


# ---------------------------------------------------------------------------
# cycles
# ---------------------------------------------------------------------------


def test_cycles_list_newest_first(seeded_client: TestClient) -> None:
    r = seeded_client.get("/cycles")
    assert r.status_code == 200
    cycles = r.json()
    assert [c["cycle_id"] for c in cycles] == ["cycle-abc123", "cycle-older"]
    # summary carries the headline fields, not the per-stage breakdown
    assert "stage_costs_usd" not in cycles[0]
    assert cycles[0]["candidates_killed"] == 2


def test_cycle_total_cost_is_verbatim_string(seeded_client: TestClient) -> None:
    cycles = seeded_client.get("/cycles").json()
    top = next(c for c in cycles if c["cycle_id"] == "cycle-abc123")
    # Decimal passes through as the exact string — not 0.1734 floated/rounded.
    assert top["total_cost_usd"] == "0.1734"
    assert isinstance(top["total_cost_usd"], str)


def test_cycle_detail_and_404(seeded_client: TestClient) -> None:
    r = seeded_client.get("/cycles/cycle-abc123")
    assert r.status_code == 200
    detail = r.json()
    assert detail["stage_costs_usd"] == {"market_context": "0.03", "critic": "0.10"}
    assert detail["outcome"] == "completed"
    assert seeded_client.get("/cycles/nope").status_code == 404


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------


def test_registry_has_killed_and_survivor_with_distinct_states(seeded_client: TestClient) -> None:
    entries = seeded_client.get("/registry").json()
    by_state = {e["state"] for e in entries}
    assert by_state == {"killed", "queued_for_approval"}
    survivor = next(e for e in entries if e["state"] == "queued_for_approval")
    # candidate params, in-sample + OOS metrics, critic verdict + concerns present
    assert survivor["candidate"]["instrument"] == "USDJPY"
    assert survivor["in_sample_evidence"]["metrics"]["sharpe_ratio"] == 2.1
    assert survivor["out_of_sample_evidence"]["metrics"]["trade_count"] == 6
    assert survivor["critic_verdict"]["verdict"] == "survive_for_now"
    items = {c["item"] for c in survivor["critic_verdict"]["concerns"]}
    assert items == {"out_of_sample_decay", "trade_count"}


def test_registry_metric_decimal_is_verbatim_string(seeded_client: TestClient) -> None:
    entries = seeded_client.get("/registry").json()
    survivor = next(e for e in entries if e["state"] == "queued_for_approval")
    assert survivor["in_sample_evidence"]["metrics"]["total_return_pct"] == "0.30"


# ---------------------------------------------------------------------------
# approval queue
# ---------------------------------------------------------------------------


def test_approval_queue_pending_with_concerns_and_report_framing(seeded_client: TestClient) -> None:
    items = seeded_client.get("/approval-queue").json()
    assert len(items) == 1
    item = items[0]
    assert item["status"] == "pending"
    assert item["critic_verdict"]["verdict"] == "survive_for_now"
    assert len(item["critic_verdict"]["concerns"]) == 2
    # reporting-agent framing attached because a CycleReport was saved
    assert "did not kill this" in item["report_explanation"]


def test_approval_queue_framing_key_always_present(seeded_client: TestClient) -> None:
    # The framing key is part of the contract (null when no report was saved);
    # the critic verdict + concerns are always present regardless.
    item = seeded_client.get("/approval-queue").json()[0]
    assert "report_explanation" in item
    assert item["critic_verdict"]["concerns"]


# ---------------------------------------------------------------------------
# backtests
# ---------------------------------------------------------------------------


def test_backtest_detail_in_sample_and_oos(seeded_client: TestClient) -> None:
    r = seeded_client.get(f"/backtests/{SURVIVOR_HASH}")
    assert r.status_code == 200
    bt = r.json()
    assert bt["config_hash"] == SURVIVOR_HASH
    assert bt["in_sample"]["metrics"]["sharpe_ratio"] == 2.1
    assert bt["out_of_sample"]["metrics"]["sharpe_ratio"] == 0.3
    assert bt["candidate"]["timeframe"] == "H1"


def test_backtest_found_by_oos_hash(seeded_client: TestClient) -> None:
    # the config_hash may be the OOS run's hash; the entry is still found
    r = seeded_client.get("/backtests/oos-survivor-hash")
    assert r.status_code == 200
    assert r.json()["in_sample"]["config_hash"] == SURVIVOR_HASH


def test_backtest_equity_curve_served_when_persisted(seeded_client: TestClient) -> None:
    bt = seeded_client.get(f"/backtests/{SURVIVOR_HASH}").json()
    assert bt["equity_curve_available"] is True
    art = bt["in_sample_artifact"]
    assert art is not None
    # net P&L is captured verbatim (a string), not recomputed by the UI
    assert art["net_pnl"] == "8200"
    assert len(art["equity_curve"]) == 5
    assert art["equity_curve"][0]["equity"] == "100000"
    # the OOS artifact is attached too
    assert bt["out_of_sample_artifact"]["net_pnl"] == "360"


def test_backtest_equity_curve_honestly_unavailable_without_artifact(
    seeded_client: TestClient,
) -> None:
    # the killed candidate has evidence but no persisted curve artifact
    bt = seeded_client.get("/backtests/is-killed-hash").json()
    assert bt["equity_curve_available"] is False
    assert bt["in_sample_artifact"] is None
    assert bt["equity_curve_notice"] == EQUITY_CURVE_NOTICE


def test_backtest_404(seeded_client: TestClient) -> None:
    assert seeded_client.get("/backtests/unknown-hash").status_code == 404


# ---------------------------------------------------------------------------
# Empty / day-one states
# ---------------------------------------------------------------------------


def test_empty_states_are_honest(empty_client: TestClient) -> None:
    assert empty_client.get("/health").status_code == 200
    assert empty_client.get("/cycles").json() == []
    assert empty_client.get("/registry").json() == []
    assert empty_client.get("/approval-queue").json() == []
    assert empty_client.get("/economics").json() == []
    assert empty_client.get("/cycles/anything").status_code == 404
    assert empty_client.get("/backtests/anything").status_code == 404
    assert empty_client.get("/economics/anything").status_code == 404


# ---------------------------------------------------------------------------
# economics
# ---------------------------------------------------------------------------


def test_economics_list_per_candidate(seeded_client: TestClient) -> None:
    rows = seeded_client.get("/economics").json()
    assert len(rows) == 2
    for r in rows:
        assert r["flag"] in {"ok", "concern", "retire"}
        assert r["in_sample"] is not None
        # win rate is surfaced together with the per-trade edge (never alone)
        assert "win_rate" in r["in_sample"]
        assert "net_expectancy_per_trade" in r["in_sample"]
        # amortized research cost is verbatim (cycle LLM cost / backtested count)
        assert r["amortized_research_cost_usd"] == "0.0867"


def test_economics_detail_has_decay_and_cost_to_edge(seeded_client: TestClient) -> None:
    r = seeded_client.get(f"/economics/{SURVIVOR_HASH}").json()
    assert r["out_of_sample"] is not None
    assert r["decay"] is not None
    assert "NOT live" in r["decay"]["note"]
    # cost-to-edge label is human-readable
    assert (
        "Broker takes" in r["in_sample"]["cost_to_edge_label"]
        or r["in_sample"]["costs_exceed_gross"]
    )


def test_economics_thin_oos_is_flagged(seeded_client: TestClient) -> None:
    # the survivor's OOS rests on 6 trades — below the statistical-power floor
    r = seeded_client.get(f"/economics/{SURVIVOR_HASH}").json()
    assert r["flag"] in {"concern", "retire"}
    assert any("statistical-power floor" in c["reason"] for c in r["concerns"])


def test_economics_404(seeded_client: TestClient) -> None:
    assert seeded_client.get("/economics/unknown-hash").status_code == 404


# ---------------------------------------------------------------------------
# universe (pair screener)
# ---------------------------------------------------------------------------


def test_universe_admitted_and_dropped_with_reasons(seeded_client: TestClient) -> None:
    u = seeded_client.get("/universe").json()
    assert u["available"] is True
    assert [a["pair"] for a in u["admitted"]] == ["EURUSD", "GBPUSD"]
    dropped = {d["pair"]: d["reason"] for d in u["dropped"]}
    assert "cost-to-move" in dropped["USDCHF"]  # dropped for cost
    assert "correlation" in dropped["NZDUSD"]  # dropped for correlation


def test_universe_correlation_matrix(seeded_client: TestClient) -> None:
    corr = seeded_client.get("/universe").json()["correlation"]
    assert corr["pairs"] == ["EURUSD", "GBPUSD"]
    assert corr["matrix"][0][0] == 1.0
    assert corr["matrix"][0][1] == 0.2


def test_universe_exposes_no_return_ranking_field(seeded_client: TestClient) -> None:
    import json

    blob = json.dumps(seeded_client.get("/universe").json()).lower()
    for forbidden in ("total_return", "profit_factor", '"pnl"', "expectancy", "sharpe", "win_rate"):
        assert forbidden not in blob, f"universe leaked a return field: {forbidden}"


def test_universe_empty_on_day_one(empty_client: TestClient) -> None:
    u = empty_client.get("/universe").json()
    assert u["available"] is False
    assert u["admitted"] == []
    assert u["dropped"] == []


def test_api_does_not_import_the_backtest_engine() -> None:
    import pathlib

    api_dir = pathlib.Path(__file__).resolve().parents[1]
    for mod in ("main.py", "store.py", "serializers.py", "config.py"):
        src = (api_dir / mod).read_text()
        imports = [ln for ln in src.splitlines() if ln.strip().startswith(("import ", "from "))]
        assert not any("BacktestEngine" in ln for ln in imports), mod
        assert not any("core.backtest" in ln for ln in imports), mod
