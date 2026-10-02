import math

import pytest

from polyglotlm.train.guard import LossSpikeMonitor, OverfitGuard


def test_guard_checkpoints_improvements_and_stops_after_patience():
    guard = OverfitGuard(["hi", "ta"], patience=2)
    assert guard.update(0, {"hi": 50, "ta": 60, "en": 10}, math.inf).improved
    assert guard.update(10, {"hi": 40, "ta": 50, "en": 10}, 3.0).improved
    worse = guard.update(20, {"hi": 41, "ta": 52, "en": 10}, 3.1)
    assert not worse.improved and not worse.stop
    assert guard.update(30, {"hi": 42, "ta": 53, "en": 10}, 3.2).stop
    assert guard.best_step == 10


def test_guard_flags_languages_whose_heldout_rises_while_train_loss_falls():
    guard = OverfitGuard(["hi", "ta"], patience=5)
    guard.update(0, {"hi": 30, "ta": 30}, 4.0)
    decision = guard.update(1, {"hi": 35, "ta": 25}, 3.0)
    assert decision.alerts == ("overfit:hi",)
    # Held-out rising together with the training loss is not overfitting.
    assert guard.update(2, {"hi": 40, "ta": 26}, 3.5).alerts == ()


def test_guard_alerts_on_english_regression_beyond_the_budget():
    guard = OverfitGuard(["hi"], patience=5, max_english_regression=3.0)
    guard.update(0, {"hi": 10}, math.inf, english_score=60.0)
    assert guard.update(1, {"hi": 9}, 2.0, english_score=57.5).alerts == ()
    assert guard.update(2, {"hi": 8}, 1.9, english_score=56.0).alerts == (
        "english_regression:4.0pts",
    )


def test_guard_requires_every_target_language():
    with pytest.raises(ValueError, match="missing"):
        OverfitGuard(["hi", "ta"], patience=1).update(0, {"hi": 1.0}, 1.0)


def test_loss_spike_monitor():
    monitor = LossSpikeMonitor(factor=2.0, warmup=3)
    assert not any(monitor.update(loss) for loss in [3.0, 2.9, 2.8, 2.7, 2.7])
    assert monitor.update(9.0)
