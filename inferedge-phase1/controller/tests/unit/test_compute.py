from unittest.mock import MagicMock, patch

import pytest

from compute import get_scheduler
from compute.local import LocalScheduler


def test_get_scheduler_local_default():
    sched = get_scheduler()
    assert isinstance(sched, LocalScheduler)


def test_get_scheduler_unknown_backend(monkeypatch):
    monkeypatch.setenv("COMPUTE_BACKEND", "kubernetes")
    with pytest.raises(ValueError, match="Unknown compute backend"):
        get_scheduler()


def test_local_scheduler_lifecycle():
    mock_ray = MagicMock()
    mock_ray.is_initialized.return_value = True
    with patch("compute.local.ray", mock_ray):
        sched = LocalScheduler()
        assert not sched.is_ready()
        sched.start()
        assert sched.is_ready()
        sched.shutdown()
        mock_ray.init.assert_called_once()
        mock_ray.shutdown.assert_called_once()


def test_local_scheduler_idempotent_start():
    mock_ray = MagicMock()
    mock_ray.is_initialized.return_value = True
    with patch("compute.local.ray", mock_ray):
        sched = LocalScheduler()
        sched.start()
        sched.start()
        mock_ray.init.assert_called_once()