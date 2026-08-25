from src.compliance.association import PersonPPEState, PPEObservation
from src.compliance.rules import ComplianceEngine
from src.config.settings import PPEClassAliases, TaxonomyConfig, ZoneConfig
from src.taxonomy import resolve_taxonomy


def _engine() -> ComplianceEngine:
    taxonomy = resolve_taxonomy(
        {0: "Hardhat", 1: "Mask", 2: "Safety Vest", 3: "Person"},
        TaxonomyConfig(
            person_labels=("Person",),
            ppe={
                "helmet": PPEClassAliases(positive=("Hardhat",), negative=()),
                "mask": PPEClassAliases(positive=("Mask",), negative=()),
                "safety_vest": PPEClassAliases(positive=("Safety Vest",), negative=()),
            },
        ),
    )
    zone = ZoneConfig(name="general", required_ppe=("helmet", "safety_vest", "mask"))
    return ComplianceEngine(zone, taxonomy)


def _obs(name: str, polarity: str = "positive") -> PPEObservation:
    return PPEObservation(
        canonical=name,
        polarity=polarity,
        confidence=0.9,
        bbox=(0, 0, 10, 10),
        label=name,
        score=1.0,
    )


def test_all_ppe_present_is_compliant():
    engine = _engine()
    person = PersonPPEState(
        person_id=17,
        bbox=(0, 0, 50, 100),
        confidence=0.9,
        observations={
            "helmet": _obs("helmet"),
            "mask": _obs("mask"),
            "safety_vest": _obs("safety_vest"),
        },
    )
    result = engine.evaluate(person)
    assert result.compliant is True
    assert result.missing_ppe == ()
    assert set(result.present_ppe) == {"helmet", "mask", "safety_vest"}


def test_helmet_missing_is_violation():
    engine = _engine()
    person = PersonPPEState(
        person_id=18,
        bbox=(0, 0, 50, 100),
        confidence=0.9,
        observations={
            "mask": _obs("mask"),
            "safety_vest": _obs("safety_vest"),
        },
    )
    result = engine.evaluate(person)
    assert result.compliant is False
    assert result.missing_ppe == ("helmet",)


def test_mask_missing_is_violation():
    engine = _engine()
    person = PersonPPEState(
        person_id=19,
        bbox=(0, 0, 50, 100),
        confidence=0.9,
        observations={"helmet": _obs("helmet"), "safety_vest": _obs("safety_vest")},
    )
    result = engine.evaluate(person)
    assert result.compliant is False
    assert result.missing_ppe == ("mask",)


def test_vest_missing_is_violation():
    engine = _engine()
    person = PersonPPEState(
        person_id=20,
        bbox=(0, 0, 50, 100),
        confidence=0.9,
        observations={"helmet": _obs("helmet"), "mask": _obs("mask")},
    )
    result = engine.evaluate(person)
    assert result.compliant is False
    assert result.missing_ppe == ("safety_vest",)


def test_multiple_missing_ppe():
    engine = _engine()
    person = PersonPPEState(person_id=21, bbox=(0, 0, 50, 100), confidence=0.8, observations={})
    result = engine.evaluate(person)
    assert result.compliant is False
    assert set(result.missing_ppe) == {"helmet", "safety_vest", "mask"}
