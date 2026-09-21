from experiments.fluidkv.calibrate_decision_costs import pair_checks


def observation(output, hit, *, prompt=1088, completed=8):
    return {"child": {"output_ids": output * completed, "meta_info": {
        "cached_tokens": hit, "prompt_tokens": prompt, "completion_tokens": completed}},
        "source_prepare": {"meta_info": {"cached_tokens": 0}},
        "owner_committed": True}


def test_calibration_separates_preserved_cached_path_from_cold_difference():
    results = {"sticky": observation([1], 1024),
               "cold-reroute": observation([2], 0),
               "prepared-full-copy": observation([1], 1024),
               "reconstruction": observation([1], 1024)}
    checks = pair_checks(results, 1024, 64, 8)
    assert checks["mechanism_valid"]
    assert not checks["cold_cached_equivalent"]
    assert checks["migration_preserves_sticky"]


def test_calibration_rejects_successful_http_without_migrated_cache_hit():
    results = {"sticky": observation([1], 1024),
               "cold-reroute": observation([1], 0),
               "prepared-full-copy": observation([1], 0),
               "reconstruction": observation([1], 1024)}
    checks = pair_checks(results, 1024, 64, 8)
    assert not checks["mechanism_valid"]
    assert checks["migration_preserves_sticky"]


def test_explicit_metadata_location_keeps_durability_and_backs_up(tmp_path):
    import json
    import sqlite3
    from types import SimpleNamespace
    from agentshift.state.schema import AgentContinuation
    from experiments.fluidkv.calibrate_decision_costs import calibration_store, finish_store

    args = SimpleNamespace(output=tmp_path / "results", metadata_directory=tmp_path / "live")
    args.output.mkdir()
    store = calibration_store(args)
    store.register_agent(AgentContinuation("saved", 0, "source", 1, (1, 2, 3)))
    assert not (args.output / "ownership.db").exists()
    finish_store(store, args)
    metadata = json.loads((args.output / "metadata_storage.json").read_text())
    assert metadata["sqlite_synchronous"] == 2  # FULL
    assert metadata["sqlite_journal_mode"] == "wal"
    assert metadata["backup_complete"]
    with sqlite3.connect(args.output / "ownership.db") as backup:
        assert backup.execute("SELECT owner_engine FROM agents WHERE agent_id='saved'").fetchone()[0] == "source"
