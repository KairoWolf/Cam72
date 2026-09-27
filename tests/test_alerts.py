from puppycam.alerts import AlertManager, Notifier, Sustained
from puppycam.analysis import FusedState
from puppycam.config import Settings


class RecordingNotifier(Notifier):
    def __init__(self):
        super().__init__()
        self.sent = []

    @property
    def enabled(self):
        return True

    def send(self, title, message, image=None, priority="high"):
        self.sent.append((title, priority, image))


def fused(ts, count, away=None, mom=True):
    return FusedState(ts=ts, count=count, best_camera="side", mom_visible=mom, away=away or {},
                      per_camera={"side": count})


def manager(**overrides):
    s = Settings(expected_puppies=9, away_seconds=60, count_low_minutes=10, mom_away_minutes=20,
                 offline_minutes=2, alert_repeat_minutes=15)
    for k, v in overrides.items():
        setattr(s, k, v)
    n = RecordingNotifier()
    m = AlertManager(s, n, snapshot=lambda: b"jpeg")
    m.started = 0.0
    return m, n


def test_sustained_needs_duration_and_tolerates_short_gaps():
    s = Sustained(10, grace=3)
    assert not s.update(0, True)
    assert not s.update(5, True)
    assert not s.update(7, False)  # short gap
    assert s.update(10, True)
    assert not s.update(20, False) and not s.update(30, False)
    assert not s.update(31, True)


def test_puppy_away_alert_fires_once_then_clears():
    m, n = manager()
    away = {"side": [4]}
    for t in range(0, 70, 5):
        m.update(float(t), fused(t, 9, away), {"Side": 0.1}, model_ready=True, mom_class=True)
    assert [a["key"] for a in m.to_dict()["active"]] == ["away"]
    assert len(n.sent) == 1 and n.sent[0][1] == "urgent" and n.sent[0][2] == b"jpeg"
    assert "#4 on side" in m.to_dict()["active"][0]["message"]
    for t in range(70, 90, 5):
        m.update(float(t), fused(t, 9), {"Side": 0.1}, model_ready=True, mom_class=True)
    assert m.to_dict()["active"] == []
    assert m.to_dict()["history"][0]["ended"] is not None


def test_count_low_alert_after_minutes_without_all_puppies():
    m, n = manager()
    m.update(0.0, fused(0, 9), {}, model_ready=True, mom_class=False)
    m.update(300.0, fused(300, 8), {}, model_ready=True, mom_class=False)
    assert not m.active
    m.update(601.0, fused(601, 8), {}, model_ready=True, mom_class=False)
    assert "count_low" in m.active
    m.update(602.0, fused(602, 9), {}, model_ready=True, mom_class=False)
    assert "count_low" not in m.active


def test_repeat_notifications_are_spaced_out():
    m, n = manager(count_low_minutes=0)
    for t in range(0, 1000, 10):
        m.update(float(t), fused(t, 5), {}, model_ready=True, mom_class=False)
    assert len(n.sent) == 2  # at t=0 and again after 15 minutes


def test_mom_away_only_with_a_mom_model():
    m, _ = manager()
    for t in (0, 1300):
        m.update(float(t), fused(t, 9, mom=False), {}, model_ready=True, mom_class=False)
    assert "mom_away" not in m.active
    m2, _ = manager()
    for t in (0, 1300):
        m2.update(float(t), fused(t, 9, mom=False), {}, model_ready=True, mom_class=True)
    assert "mom_away" in m2.active


def test_camera_offline_alert():
    m, n = manager()
    m.update(10.0, None, {"Side": 150.0, "Top": 1.0}, model_ready=False, mom_class=False)
    assert list(m.active) == ["offline:Side"]
    m.update(20.0, None, {"Side": 0.5, "Top": 1.0}, model_ready=False, mom_class=False)
    assert not m.active


def test_no_model_means_no_puppy_alerts():
    m, n = manager(count_low_minutes=0)
    m.update(0.0, None, {"Side": 0.1}, model_ready=False, mom_class=False)
    assert not m.active and not n.sent
