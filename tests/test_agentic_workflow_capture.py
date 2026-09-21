from experiments.agentic_profile.capture_workflow import explicit_already_stopped, first_telemetry


def test_partial_telemetry_line_is_not_readiness(tmp_path):
    path = tmp_path/'telemetry.jsonl'
    path.write_text('{"event":"metadata","monotonic":0}\n')
    assert first_telemetry(path) is None
    path.write_text('{"event":"metadata","monotonic":0}\n{"event":"sample","monotonic":123')
    assert first_telemetry(path) is None
    path.write_text('{"event":"metadata","monotonic":0}\n{"event":"sample","monotonic":123}\n')
    assert first_telemetry(path) == {'event': 'sample', 'monotonic': 123}


def test_only_explicit_not_profiling_is_tolerated():
    assert explicit_already_stopped('Profiling is not in progress. Call /start_profile first.')
    assert not explicit_already_stopped('HTTP500 internal server error')
    assert not explicit_already_stopped('profile failed')
