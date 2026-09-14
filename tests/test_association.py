from src.compliance.association import PPEObservation, associate_ppe, resolve_ppe_conflict
from src.config.settings import AssociationConfig, PPEClassAliases, TaxonomyConfig
from src.inference.result import Detection
from src.taxonomy import resolve_taxonomy
from src.tracking.tracker import TrackedDetection


def _taxonomy():
    names = {0: "Hardhat", 1: "Person", 2: "NO-Hardhat", 3: "Safety Vest"}
    return resolve_taxonomy(
        names,
        TaxonomyConfig(
            person_labels=("Person",),
            ppe={
                "helmet": PPEClassAliases(positive=("Hardhat",), negative=("NO-Hardhat",)),
                "safety_vest": PPEClassAliases(positive=("Safety Vest",), negative=()),
            },
        ),
    )


def _person(track_id: int, bbox, conf=0.9) -> TrackedDetection:
    det = Detection(class_id=1, label="Person", confidence=conf, bbox=bbox, track_id=track_id)
    return TrackedDetection(detection=det, track_id=track_id)


def test_helmet_assigned_to_correct_person():
    taxonomy = _taxonomy()
    persons = [
        _person(17, (0, 0, 100, 200)),
        _person(18, (300, 0, 400, 200)),
    ]
    detections = [
        Detection(0, "Hardhat", 0.94, (20, 5, 60, 40)),
        Detection(1, "Person", 0.9, (0, 0, 100, 200)),
        Detection(1, "Person", 0.9, (300, 0, 400, 200)),
    ]
    states = associate_ppe(persons, detections, taxonomy, AssociationConfig())
    by_id = {item.person_id: item for item in states}
    assert by_id[17].status_for("helmet") == "present"
    assert by_id[18].status_for("helmet") == "not_associated"


def test_helmet_not_assigned_to_unrelated_person():
    taxonomy = _taxonomy()
    persons = [_person(1, (0, 0, 80, 160)), _person(2, (200, 0, 280, 160))]
    detections = [Detection(0, "Hardhat", 0.9, (210, 5, 250, 40))]
    states = associate_ppe(persons, detections, taxonomy, AssociationConfig())
    by_id = {item.person_id: item for item in states}
    assert by_id[1].status_for("helmet") == "not_associated"
    assert by_id[2].status_for("helmet") == "present"


def test_overlapping_workers_get_nearest_helmet():
    taxonomy = _taxonomy()
    persons = [
        _person(1, (0, 0, 120, 200)),
        _person(2, (80, 0, 200, 200)),
    ]
    detections = [Detection(0, "Hardhat", 0.91, (90, 8, 110, 36))]
    states = associate_ppe(persons, detections, taxonomy, AssociationConfig())
    assigned = [item.person_id for item in states if item.status_for("helmet") == "present"]
    assert len(assigned) == 1
    assert assigned[0] in {1, 2}


def _full_taxonomy():
    names = {
        0: "Hardhat",
        1: "Person",
        2: "NO-Hardhat",
        3: "Safety Vest",
        4: "Mask",
        5: "NO-Mask",
        6: "NO-Safety Vest",
    }
    return resolve_taxonomy(
        names,
        TaxonomyConfig(
            person_labels=("Person",),
            ppe={
                "helmet": PPEClassAliases(positive=("Hardhat",), negative=("NO-Hardhat",)),
                "mask": PPEClassAliases(positive=("Mask",), negative=("NO-Mask",)),
                "safety_vest": PPEClassAliases(positive=("Safety Vest",), negative=("NO-Safety Vest",)),
            },
        ),
    )


def test_mask_and_vest_associated_to_same_person():
    taxonomy = _full_taxonomy()
    persons = [_person(7, (0, 0, 100, 200))]
    detections = [
        Detection(0, "Hardhat", 0.9, (20, 5, 60, 40)),
        Detection(4, "Mask", 0.88, (35, 30, 55, 55)),
        Detection(3, "Safety Vest", 0.91, (15, 50, 85, 140)),
    ]
    state = associate_ppe(persons, detections, taxonomy, AssociationConfig())[0]
    assert state.status_for("helmet") == "present"
    assert state.status_for("mask") == "present"
    assert state.status_for("safety_vest") == "present"


def test_person_with_no_ppe_is_not_associated():
    taxonomy = _full_taxonomy()
    persons = [_person(3, (0, 0, 80, 160))]
    states = associate_ppe(persons, [], taxonomy, AssociationConfig())
    assert states[0].status_for("helmet") == "not_associated"
    assert states[0].status_for("mask") == "not_associated"
    assert states[0].status_for("safety_vest") == "not_associated"


def test_ppe_far_from_person_is_not_assigned():
    taxonomy = _full_taxonomy()
    persons = [_person(1, (0, 0, 80, 160))]
    detections = [Detection(0, "Hardhat", 0.95, (500, 5, 540, 40))]
    state = associate_ppe(persons, detections, taxonomy, AssociationConfig())[0]
    assert state.status_for("helmet") == "not_associated"


def test_no_persons_yields_empty_association():
    taxonomy = _full_taxonomy()
    detections = [Detection(0, "Hardhat", 0.9, (20, 5, 60, 40))]
    assert associate_ppe([], detections, taxonomy, AssociationConfig()) == []


def test_negative_helmet_marks_missing():
    taxonomy = _full_taxonomy()
    persons = [_person(1, (0, 0, 100, 200))]
    detections = [Detection(2, "NO-Hardhat", 0.87, (20, 5, 60, 40))]
    state = associate_ppe(persons, detections, taxonomy, AssociationConfig())[0]
    assert state.status_for("helmet") == "missing"
    assert state.helmet_state() == "MISSING"


def test_negative_mask_marks_missing():
    taxonomy = _full_taxonomy()
    persons = [_person(4, (0, 0, 100, 200))]
    detections = [Detection(5, "NO-Mask", 0.9, (35, 30, 55, 55))]
    state = associate_ppe(persons, detections, taxonomy, AssociationConfig())[0]
    assert state.mask_state() == "MISSING"


def test_negative_vest_marks_missing():
    taxonomy = _full_taxonomy()
    persons = [_person(5, (0, 0, 100, 200))]
    detections = [Detection(6, "NO-Safety Vest", 0.9, (15, 50, 85, 140))]
    state = associate_ppe(persons, detections, taxonomy, AssociationConfig())[0]
    assert state.vest_state() == "MISSING"


def test_all_negative_ppe_on_one_person():
    taxonomy = _full_taxonomy()
    persons = [_person(6, (0, 0, 100, 200))]
    detections = [
        Detection(2, "NO-Hardhat", 0.9, (20, 5, 60, 40)),
        Detection(5, "NO-Mask", 0.88, (35, 30, 55, 55)),
        Detection(6, "NO-Safety Vest", 0.91, (15, 50, 85, 140)),
    ]
    state = associate_ppe(persons, detections, taxonomy, AssociationConfig())[0]
    assert state.helmet_state() == "MISSING"
    assert state.mask_state() == "MISSING"
    assert state.vest_state() == "MISSING"


def test_two_people_different_ppe():
    taxonomy = _full_taxonomy()
    persons = [_person(17, (0, 0, 100, 200)), _person(21, (220, 0, 320, 200))]
    detections = [
        Detection(0, "Hardhat", 0.9, (20, 5, 60, 40)),
        Detection(4, "Mask", 0.88, (35, 30, 55, 55)),
        Detection(3, "Safety Vest", 0.91, (15, 50, 85, 140)),
        Detection(2, "NO-Hardhat", 0.86, (240, 5, 280, 40)),
        Detection(4, "Mask", 0.84, (250, 28, 275, 52)),
        Detection(6, "NO-Safety Vest", 0.9, (230, 50, 310, 140)),
    ]
    states = associate_ppe(persons, detections, taxonomy, AssociationConfig())
    by_id = {item.person_id: item for item in states}
    assert by_id[17].helmet_state() == "PRESENT"
    assert by_id[17].mask_state() == "PRESENT"
    assert by_id[17].vest_state() == "PRESENT"
    assert by_id[21].helmet_state() == "MISSING"
    assert by_id[21].mask_state() == "PRESENT"
    assert by_id[21].vest_state() == "MISSING"


def test_three_people_different_ppe():
    taxonomy = _full_taxonomy()
    persons = [
        _person(1, (0, 0, 90, 200)),
        _person(2, (120, 0, 210, 200)),
        _person(3, (240, 0, 330, 200)),
    ]
    detections = [
        Detection(0, "Hardhat", 0.9, (20, 5, 55, 40)),
        Detection(4, "Mask", 0.88, (30, 28, 50, 50)),
        Detection(3, "Safety Vest", 0.9, (10, 50, 80, 140)),
        Detection(2, "NO-Hardhat", 0.87, (140, 5, 175, 38)),
        Detection(4, "Mask", 0.85, (150, 28, 170, 50)),
        Detection(3, "Safety Vest", 0.86, (130, 50, 200, 140)),
        Detection(0, "Hardhat", 0.91, (260, 5, 295, 38)),
        Detection(5, "NO-Mask", 0.84, (270, 28, 290, 50)),
        Detection(6, "NO-Safety Vest", 0.88, (250, 50, 320, 140)),
    ]
    states = associate_ppe(persons, detections, taxonomy, AssociationConfig())
    by_id = {item.person_id: item for item in states}
    assert by_id[1].helmet_state() == "PRESENT" and by_id[1].mask_state() == "PRESENT"
    assert by_id[2].helmet_state() == "MISSING" and by_id[2].vest_state() == "PRESENT"
    assert by_id[3].helmet_state() == "PRESENT" and by_id[3].mask_state() == "MISSING"
    assert by_id[3].vest_state() == "MISSING"


def test_ppe_near_wrong_person_stays_unassigned():
    taxonomy = _full_taxonomy()
    persons = [_person(1, (0, 0, 80, 160)), _person(2, (400, 0, 480, 160))]
    detections = [Detection(0, "Hardhat", 0.92, (90, 5, 120, 35))]
    states = associate_ppe(persons, detections, taxonomy, AssociationConfig())
    by_id = {item.person_id: item for item in states}
    assert by_id[1].helmet_state() == "UNKNOWN"
    assert by_id[2].helmet_state() == "UNKNOWN"


def test_conflicting_hardhat_prefers_higher_confidence():
    taxonomy = _full_taxonomy()
    persons = [_person(9, (0, 0, 100, 200))]
    detections = [
        Detection(0, "Hardhat", 0.92, (20, 5, 60, 40)),
        Detection(2, "NO-Hardhat", 0.40, (22, 6, 58, 38)),
    ]
    state = associate_ppe(persons, detections, taxonomy, AssociationConfig())[0]
    assert state.status_for("helmet") == "present"
    assert state.helmet_state() == "PRESENT"


def test_conflicting_hardhat_prefers_stronger_negative():
    taxonomy = _full_taxonomy()
    persons = [_person(9, (0, 0, 100, 200))]
    detections = [
        Detection(0, "Hardhat", 0.35, (20, 5, 60, 40)),
        Detection(2, "NO-Hardhat", 0.93, (22, 6, 58, 38)),
    ]
    state = associate_ppe(persons, detections, taxonomy, AssociationConfig())[0]
    assert state.status_for("helmet") == "missing"


def test_resolve_conflict_tie_prefers_positive():
    pos = PPEObservation("helmet", "positive", 0.80, (20, 5, 60, 40), "Hardhat", 0.5)
    neg = PPEObservation("helmet", "negative", 0.80, (22, 6, 58, 38), "NO-Hardhat", 0.5)
    chosen = resolve_ppe_conflict([pos, neg], AssociationConfig(prefer_positive_on_tie=True), person_id=1)
    assert chosen.polarity == "positive"


def test_empty_detections_leave_ppe_unknown():
    taxonomy = _full_taxonomy()
    persons = [_person(8, (0, 0, 80, 160))]
    state = associate_ppe(persons, [], taxonomy, AssociationConfig())[0]
    assert state.helmet_state() == "UNKNOWN"
    assert state.mask_state() == "UNKNOWN"
    assert state.vest_state() == "UNKNOWN"
    assert state.person_id == 8
