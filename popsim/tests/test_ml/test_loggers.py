from types import SimpleNamespace

import pytest

from popsim.ml.loggers import WandbLogger


class _FakeInterface:
    def __init__(self, run_should_stop, broken=False):
        self.run_should_stop = run_should_stop
        self.broken = broken

    def deliver_stop_status(self):
        if self.broken:
            raise AttributeError("internal API changed")
        result = SimpleNamespace(
            response=SimpleNamespace(stop_status_response=SimpleNamespace(run_should_stop=self.run_should_stop))
        )
        return SimpleNamespace(wait_or=lambda timeout: result)


class _FakeRun:
    def __init__(self, run_should_stop, broken=False):
        self._interface = _FakeInterface(run_should_stop, broken)
        self.logged = []

    def define_metric(self, *args, **kwargs):
        pass

    def log(self, dictionary):
        self.logged.append(dictionary)


def _make_logger(run):
    logger = WandbLogger(run)
    # Force the stop poll to run on every log call.
    logger.stop_poll_interval_s = 0.0
    return logger


def test_wandb_logger_raises_on_stop_request():
    logger = _make_logger(_FakeRun(run_should_stop=True))
    with pytest.raises(KeyboardInterrupt):
        logger.log({"train/loss": 1.0})


def test_wandb_logger_continues_when_no_stop_request():
    run = _FakeRun(run_should_stop=False)
    logger = _make_logger(run)
    logger.log({"train/loss": 1.0})
    assert len(run.logged) == 1


def test_wandb_logger_tolerates_internal_api_changes():
    run = _FakeRun(run_should_stop=True, broken=True)
    logger = _make_logger(run)
    logger.log({"train/loss": 1.0})
    assert len(run.logged) == 1


def test_wandb_logger_rate_limits_stop_polls():
    run = _FakeRun(run_should_stop=True)
    logger = WandbLogger(run)
    # With a long poll interval, the stop check should not fire right after construction.
    logger.stop_poll_interval_s = 3600.0
    logger.log({"train/loss": 1.0})
    assert len(run.logged) == 1
