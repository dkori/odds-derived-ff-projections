"""Flask playground for exploring stored projection distributions.

A local dev tool (not the production front end) to eyeball the charts before
committing to a hosted DB. It reads **only** through :class:`ProjectionsStore`,
so swapping the file-backed reader for a DigitalOcean DB reader later needs no
change here.

Run:
    pip install -e ".[web]"
    python -m ffmc.web            # or: flask --app ffmc.web run --debug
"""
from __future__ import annotations

import json
import os

from flask import Flask, jsonify, render_template, request

from ffmc.sim.store import ProjectionsStore
from ffmc.sim.viz import build_figure


def create_app(store_dir: str | None = None) -> Flask:
    app = Flask(__name__)
    store = ProjectionsStore(store_dir or os.environ.get("FFMC_STORE_DIR", "data/projections"))

    @app.get("/")
    def index():
        weeks = store.list_weeks()
        return render_template("index.html", weeks=weeks)

    @app.get("/api/weeks")
    def api_weeks():
        return jsonify(store.list_weeks())

    @app.get("/api/snapshots/<week>")
    def api_snapshots(week: str):
        return jsonify(store.list_snapshots(week))

    @app.get("/api/players")
    def api_players():
        """Player names for a week/snapshot, optionally scoped to a group.

        ``group`` is ``qb``, ``skill``, or ``all`` (backs the combobox datalist
        for the QB vs WR/RB/TE comparison views).
        """
        week = request.args.get("week", "")
        snapshot = request.args.get("snapshot") or None
        group = request.args.get("group") or None
        if not week:
            return jsonify([])
        try:
            return jsonify(store.player_names(week, snapshot, group=group))
        except FileNotFoundError:
            return jsonify([])

    @app.get("/api/search")
    def api_search():
        week = request.args.get("week", "")
        query = request.args.get("q", "")
        snapshot = request.args.get("snapshot") or None
        group = request.args.get("group") or None
        if not week:
            return jsonify([])
        try:
            return jsonify(
                store.search_players(
                    week, query, snapshot_id=snapshot, limit=12, group=group
                )
            )
        except FileNotFoundError:
            return jsonify([])

    @app.get("/api/figure")
    def api_figure():
        """Return a Plotly figure (JSON) for the selected players."""
        import plotly.utils

        from ffmc.sim.viz import data_notes

        week = request.args.get("week", "")
        snapshot = request.args.get("snapshot") or None
        mode = request.args.get("mode", "density")
        if mode not in ("density", "cdf"):
            mode = "density"
        names = [n for n in request.args.get("players", "").split(",") if n.strip()]
        if not week or not names:
            return jsonify({"error": "week and players required"}), 400

        snap = store.load(week, snapshot)
        players = snap.get("players", {})
        series, missing, notes = [], [], {}
        for name in names:
            rec = players.get(name)
            if rec:
                series.append((name, rec))
                notes[name] = data_notes(rec)
            else:
                missing.append(name)
        if not series:
            return jsonify({"error": "no known players", "missing": missing}), 404

        fig = build_figure(snap, series, mode=mode)
        fig_json = json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))
        return jsonify(
            {"figure": fig_json, "missing": missing, "n": len(series), "notes": notes}
        )

    return app
