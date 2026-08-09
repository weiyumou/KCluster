"""Tests for resuming a Vertex launch and for its throughput reporting.

Both exist to stop a relaunch costing money twice: ``collected_inputs`` decides
what NOT to pay for again, and ``job_rate`` is the number you size the next
launch from. Neither touches GCP here — the result lookup is stubbed.
"""

import datetime
import json

import pytest

pytest.importorskip("google.cloud.aiplatform")

from kcluster.commands import vertex_launch  # noqa: E402
from kcluster.engine.vertex import job_rate  # noqa: E402


def _jobs_file(tmp_path, entries):
    path = tmp_path / "launched_jobs.jsonl"
    with open(path, "w") as f:
        for job_id, data_path in entries:
            f.write(json.dumps({"job_id": job_id, "data_path": data_path,
                                "resource_name": f"projects/p/locations/l/batchPredictionJobs/{job_id}"}) + "\n")
    return str(path)


@pytest.fixture
def stub_results(monkeypatch):
    """Pretend a given set of job ids have collected pmi.npy in GCS."""
    def _stub(with_results):
        monkeypatch.setattr(vertex_launch, "download_pmi",
                            lambda job_id, config: object() if job_id in with_results else None)
    return _stub


def test_no_job_log_means_nothing_is_skipped(tmp_path, stub_results):
    stub_results(set())
    assert vertex_launch.collected_inputs(str(tmp_path / "absent.jsonl"), config=None) == {}


def test_only_jobs_with_results_count_as_done(tmp_path, stub_results):
    jobs = _jobs_file(tmp_path, [("bio_1", "data/Bio.jsonl"), ("chem_1", "data/Chem.jsonl")])
    # chem was launched but never produced a matrix (failed, cancelled, running).
    stub_results({"bio_1"})
    assert vertex_launch.collected_inputs(jobs, config=None) == {"data/Bio.jsonl": "bio_1"}


def test_a_relaunch_supersedes_an_earlier_failed_attempt(tmp_path, stub_results):
    # The course was tried, failed, and tried again; the later job has results.
    jobs = _jobs_file(tmp_path, [("bio_1", "data/Bio.jsonl"), ("bio_2", "data/Bio.jsonl")])
    stub_results({"bio_2"})
    assert vertex_launch.collected_inputs(jobs, config=None) == {"data/Bio.jsonl": "bio_2"}


def test_blank_lines_in_the_job_log_are_tolerated(tmp_path, stub_results):
    path = tmp_path / "launched_jobs.jsonl"
    path.write_text(json.dumps({"job_id": "bio_1", "data_path": "data/Bio.jsonl"}) + "\n\n")
    stub_results({"bio_1"})
    assert vertex_launch.collected_inputs(str(path), config=None) == {"data/Bio.jsonl": "bio_1"}


# --- throughput reporting --------------------------------------------------
class _Job:
    def __init__(self, start=None, end=None):
        if start is not None:
            self.start_time = start
        if end is not None:
            self.end_time = end


def test_rate_uses_vertex_start_and_end_not_wall_clock():
    start = datetime.datetime(2026, 8, 9, 12, 0, 0)
    job = _Job(start, start + datetime.timedelta(minutes=10))
    out = job_rate(job, 120_000)
    assert "10.0 min" in out and "120,000 instances" in out and "200 instances/s" in out


def test_rate_without_an_instance_count_still_reports_duration():
    start = datetime.datetime(2026, 8, 9, 12, 0, 0)
    assert job_rate(_Job(start, start + datetime.timedelta(minutes=3)), None) == "ran 3.0 min"


@pytest.mark.parametrize("job", [
    _Job(),                                                                  # no timestamps at all
    _Job(datetime.datetime(2026, 8, 9, 12, 0, 0), None),                     # still running
    _Job(datetime.datetime(2026, 8, 9, 12, 0, 0),
         datetime.datetime(2026, 8, 9, 12, 0, 0)),                           # zero duration
])
def test_rate_is_empty_when_it_cannot_be_computed(job):
    # An unreportable rate must not crash the wait loop or print a fake number.
    assert job_rate(job, 1000) == ""


# --- stall detection -------------------------------------------------------
class _Stats:
    def __init__(self, successful_count):
        self.successful_count = successful_count


class _Resource:
    def __init__(self, stats):
        self.completion_stats = stats


class _RunningJob:
    """A job stuck in RUNNING whose progress counter the test drives."""
    def __init__(self, counts):
        self.name = "job-1"
        self.display_name = "stuck-course"
        self.resource_name = "projects/p/locations/l/batchPredictionJobs/1"
        self._counts = list(counts)
        self.state = 3  # JOB_STATE_RUNNING
        self._gca_resource = _Resource(_Stats(self._counts[0]))

    def advance(self):
        if len(self._counts) > 1:
            self._counts.pop(0)
        self._gca_resource.completion_stats = _Stats(self._counts[0])


def test_instances_done_reads_the_live_counter():
    from kcluster.engine.vertex import instances_done
    assert instances_done(_RunningJob([16816])) == 16816


def test_instances_done_is_zero_when_the_job_never_dispatched():
    # A job that has not started dispatching leaves completion_stats unset;
    # that is precisely the state the stall check has to notice.
    from kcluster.engine.vertex import instances_done

    class _NoStats:
        _gca_resource = _Resource(None)
    assert instances_done(_NoStats()) == 0


def _run_wait(monkeypatch, job, clock, stall_after, polls):
    """Drive wait_for_job_completion for a fixed number of polls."""
    import kcluster.engine.vertex as vx

    warnings = []
    monkeypatch.setattr(vx.logger, "warning", lambda m: warnings.append(m))
    monkeypatch.setattr(vx.logger, "info", lambda m: None)
    monkeypatch.setattr(vx.time, "monotonic", lambda: clock[0])

    def fake_sleep(_):
        clock[0] += 600          # ten minutes of wall clock per poll
        job.advance()
        if clock[0] > 600 * polls:
            raise StopIteration  # the job never reaches a terminal state, so stop the loop here
    monkeypatch.setattr(vx.time, "sleep", fake_sleep)
    try:
        vx.wait_for_job_completion([{"job_obj": job, "job_id": "j", "num_questions": 1, "num_instances": 100_000}],
                                   config=None, stall_after_seconds=stall_after)
    except StopIteration:
        pass
    return warnings


def test_a_job_that_never_dispatches_is_reported_stalled(monkeypatch):
    job = _RunningJob([0])            # counter never moves
    warnings = _run_wait(monkeypatch, job, [0.0], stall_after=1800, polls=6)
    assert any("STALLED" in w for w in warnings), "a zero counter for an hour was not flagged"
    assert sum("STALLED" in w for w in warnings) == 1, "the warning should fire once, not every poll"


def test_a_progressing_job_is_never_flagged(monkeypatch):
    job = _RunningJob([0, 5_000, 12_000, 20_000, 31_000, 44_000, 60_000])
    warnings = _run_wait(monkeypatch, job, [0.0], stall_after=1800, polls=6)
    assert not any("STALLED" in w for w in warnings)


def test_a_job_that_dies_mid_run_is_flagged(monkeypatch):
    # Dispatches, then the counter freezes — a replica lost after starting.
    job = _RunningJob([1_000, 9_000, 9_000, 9_000, 9_000, 9_000, 9_000])
    warnings = _run_wait(monkeypatch, job, [0.0], stall_after=1800, polls=6)
    assert any("STALLED" in w for w in warnings)


def test_the_check_can_be_disabled(monkeypatch):
    job = _RunningJob([0])
    assert not _run_wait(monkeypatch, job, [0.0], stall_after=0, polls=6)
