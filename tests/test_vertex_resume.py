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
