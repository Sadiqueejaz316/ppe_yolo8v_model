from src.compliance.association import PPEObservation, PersonPPEState
from src.compliance.stability import PPEStateStabilizer


def _obs(name: str, polarity: str = "positive") -> PPEObservation:
    return PPEObservation(
        canonical=name,
        polarity=polarity,
        confidence=0.9,
        bbox=(20, 5, 60, 40),
        label=name,
        score=1.0,
    )


def _person(person_id: int, helmet: str) -> PersonPPEState:
    observations = {}
    if helmet == "present":
        observations["helmet"] = _obs("helmet", "positive")
    elif helmet == "missing":
        observations["helmet"] = _obs("helmet", "negative")
    return PersonPPEState(person_id=person_id, bbox=(0, 0, 80, 160), confidence=0.9, observations=observations)


def test_first_observation_is_accepted_immediately():
    stab = PPEStateStabilizer(stable_frames=3)
    out = stab.update([_person(17, "missing")])
    assert out[0].status_for("helmet") == "missing"


def test_single_flicker_does_not_change_displayed_state():
    stab = PPEStateStabilizer(stable_frames=3)
    stab.update([_person(17, "missing")])
    flipped = stab.update([_person(17, "present")])
    assert flipped[0].status_for("helmet") == "missing"
    flipped = stab.update([_person(17, "missing")])
    assert flipped[0].status_for("helmet") == "missing"


def test_consistent_frames_commit_new_state():
    stab = PPEStateStabilizer(stable_frames=3)
    stab.update([_person(17, "missing")])
    stab.update([_person(17, "present")])
    stab.update([_person(17, "present")])
    committed = stab.update([_person(17, "present")])
    assert committed[0].status_for("helmet") == "present"


def test_disappeared_person_is_forgotten():
    stab = PPEStateStabilizer(stable_frames=3)
    stab.update([_person(17, "present")])
    assert stab.update([]) == []
    back = stab.update([_person(17, "missing")])
    assert back[0].status_for("helmet") == "missing"


def test_stable_frames_one_is_passthrough():
    stab = PPEStateStabilizer(stable_frames=1)
    first = stab.update([_person(3, "present")])
    second = stab.update([_person(3, "missing")])
    assert first[0].status_for("helmet") == "present"
    assert second[0].status_for("helmet") == "missing"
