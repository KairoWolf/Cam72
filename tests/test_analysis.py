from puppycam.analysis import CameraAnalyzer, find_away, fuse
from puppycam.detector import MOM
from puppycam.tracking import PuppyTracker

from .conftest import det


def litter(n, x0=100, y0=100, w=60, h=40, gap=5):
    return [det([x0 + i * (w + gap), y0, x0 + i * (w + gap) + w, y0 + h]) for i in range(n)]


def test_find_away_flags_puppy_far_from_pile():
    pups = litter(4)
    loner = det([900, 600, 960, 640])
    away = find_away(pups + [loner], None, factor=0.75)
    assert away == [loner]


def test_find_away_uses_mom_as_the_main_group():
    mom = det([80, 80, 400, 300], cls=MOM)
    pair = litter(2, x0=900, y0=600)  # two puppies together, but away from mom
    near_mom = litter(1, x0=150, y0=150)
    away = find_away(pair + near_mom, mom, factor=0.75)
    assert away == pair


def test_find_away_needs_something_to_compare_with():
    assert find_away(litter(1), None, 0.75) == []
    assert find_away([], None, 0.75) == []


def test_analyzer_clamps_to_litter_size_and_keeps_best():
    analyzer = CameraAnalyzer("cam", expected=3)
    dets = litter(4)
    for d, c in zip(dets, (0.9, 0.8, 0.45, 0.7)):
        d.conf = c
    state = analyzer.update(dets, ts=0.0, conf=0.4)
    assert state.count == 3
    assert sorted(p.conf for p in state.puppies) == [0.7, 0.8, 0.9]
    assert "low-confidence puppy" in state.uncertain  # the dropped extra is worth labeling


def test_analyzer_hysteresis_keeps_partly_hidden_puppy():
    analyzer = CameraAnalyzer("cam", expected=9)
    dets = litter(3)
    analyzer.update(dets, ts=0.0, conf=0.5)
    fading = litter(3)
    fading[1].conf = 0.3  # below the threshold, but it was there a moment ago
    state = analyzer.update(fading, ts=0.25, conf=0.5)
    assert state.count == 3
    new = litter(4)
    new[3].conf = 0.3  # a brand-new weak detection does not count
    assert analyzer.update(new, ts=0.5, conf=0.5).count == 3


def test_hysteresis_does_not_double_count_a_weak_duplicate():
    analyzer = CameraAnalyzer("cam", expected=9)
    analyzer.update(litter(2), ts=0.0, conf=0.5)
    again = litter(2)
    duplicate = det(list(again[0].box), conf=0.3)  # weaker second box on the same puppy
    assert analyzer.update(again + [duplicate], ts=0.25, conf=0.5).count == 2


def test_analyzer_smoothing_ignores_single_frame_flicker():
    analyzer = CameraAnalyzer("cam", expected=9, smooth_seconds=5)
    for i in range(8):
        analyzer.update(litter(5), ts=i * 0.25, conf=0.4)
    state = analyzer.update(litter(4), ts=2.0, conf=0.4)
    assert state.count == 4
    assert state.smoothed == 5
    assert "count flickering" in state.uncertain


def test_analyzer_numbers_are_stable():
    analyzer = CameraAnalyzer("cam", expected=9)
    first = analyzer.update(litter(3), ts=0, conf=0.4)
    numbers = {tuple(p.box): p.number for p in first.puppies}
    shuffled = list(reversed(litter(3)))
    second = analyzer.update(shuffled, ts=0.25, conf=0.4)
    assert {tuple(p.box): p.number for p in second.puppies} == numbers
    assert sorted(numbers.values()) == [1, 2, 3]


def test_tracker_reuses_free_numbers_and_forgets_old_tracks():
    tracker = PuppyTracker(max_age=5)
    assert tracker.update([[0, 0, 10, 10], [50, 0, 60, 10]], ts=0) == [1, 2]
    assert tracker.update([[50, 0, 60, 10]], ts=1) == [2]
    # puppy 1 has been gone for longer than max_age: its number is free again
    assert tracker.update([[50, 0, 60, 10], [200, 200, 210, 210]], ts=5.5) == [2, 1]
    # everything forgotten after a long gap
    assert tracker.update([[400, 400, 410, 410]], ts=60) == [1]


def test_fuse_takes_the_best_view():
    a = CameraAnalyzer("top", expected=9).update(litter(6), ts=0, conf=0.4)
    b = CameraAnalyzer("side", expected=9).update(litter(9) + [det([0, 0, 900, 900], cls=MOM)], ts=0, conf=0.4)
    fused = fuse([a, b], ts=0)
    assert fused.count == 9
    assert fused.best_camera == "side"
    assert fused.mom_visible
    assert fused.per_camera == {"top": 6, "side": 9}
    assert fuse([], ts=0).count == 0
