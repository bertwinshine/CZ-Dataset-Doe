from cz_doe.certify import certify


def failed(checks):
    return {c.name for c in checks if not c.passed}


def test_consistent_field_passes(field, anchor):
    assert failed(certify(field, anchor)) == set()


def test_wrong_hot_wall_temperature_is_rejected(field, anchor):
    hotter = anchor.__class__(anchor.crystal_rpm, anchor.crucible_rpm, anchor.t_hot + 5)
    assert {"free surface T", "crucible bottom T", "T bounds"} <= failed(certify(field, hotter))


def test_free_surface_swirl_frozen_at_baseline_is_rejected(field, anchor):
    # The v1 crucible sweep failed exactly this way.
    faster = anchor.__class__(anchor.crystal_rpm, -6.0, anchor.t_hot)
    assert "free surface swirl" in failed(certify(field, faster))


def test_transposed_velocity_columns_are_rejected(field, anchor):
    swapped = field.rename(columns={"u_r": "u_z", "u_z": "u_r"})[field.columns]
    assert "axis u_r -> 0" in failed(certify(swapped, anchor))


def test_swapped_coordinate_columns_are_rejected(field, anchor):
    swapped = field.rename(columns={"r": "z", "z": "r"})[field.columns]
    assert "domain extent" in failed(certify(swapped, anchor))
