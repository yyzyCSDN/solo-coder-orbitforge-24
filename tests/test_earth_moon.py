import math

import pytest

from orbitforge.core.constants import DAY_S, MU_EARTH_KM3_S2, MU_MOON_KM3_S2, MOON_SEMIMAJOR_KM
from orbitforge.core.errors import NoSolutionError, ValidationError
from orbitforge.interplanetary.earth_moon import (
    CaptureOrbit,
    EarthMoonTransferDesigner,
    ParkingOrbit,
)


def _designer():
    return EarthMoonTransferDesigner(ParkingOrbit(185.0), CaptureOrbit(110.0, 310.0))


def test_soi_boundary_derived_not_hardcoded():
    d = _designer()
    expected = MOON_SEMIMAJOR_KM * (MU_MOON_KM3_S2 / MU_EARTH_KM3_S2) ** 0.4
    assert d.soi_radius_km == pytest.approx(expected, rel=1e-12)
    assert 60000.0 < d.soi_radius_km < 70000.0
    assert d.soi_entry_geocentric_km == pytest.approx(MOON_SEMIMAJOR_KM - d.soi_radius_km)


def test_apollo_like_design_ranges():
    d = _designer().from_total_tof(3.05)
    assert -2.5 < d.earth_leg.c3_km2_s2 < -1.0
    assert 3.0 < d.earth_leg.departure_dv_km_s < 3.3
    assert 0.7 < d.soi_patch.v_inf_km_s < 1.3
    assert 0.6 < d.moon_leg.capture_dv_km_s < 1.1
    assert d.total_dv_km_s < 4.5
    assert 2.0 < d.moon_leg.capture_period_h < 2.5


def test_c3_tof_roundtrip():
    d = _designer()
    ref = d.from_c3(-1.7)
    back = d.from_total_tof(ref.total_tof_days)
    assert back.earth_leg.c3_km2_s2 == pytest.approx(ref.earth_leg.c3_km2_s2, rel=1e-8)


def test_higher_c3_shorter_tof():
    d = _designer()
    slow = d.from_c3(-1.9)
    fast = d.from_c3(-0.5)
    assert fast.total_tof_days < slow.total_tof_days
    assert fast.earth_leg.departure_dv_km_s > slow.earth_leg.departure_dv_km_s


def test_min_energy_arrives_tangent_at_soi():
    d = _designer().min_energy()
    assert d.earth_leg.conic_kind == "ellipse"
    # 远地点恰在 SOI 进入点：径向速度相对总速度可忽略（相切到达）
    assert abs(d.soi_patch.v_sc_radial_km_s) < 1e-3 * d.earth_leg.soi_entry_speed_km_s
    assert d.earth_leg.c3_km2_s2 == pytest.approx(_designer().min_c3_km2_s2, abs=1e-6)


def test_soi_patch_is_explicit_frame_change():
    d = _designer().from_c3(-1.5)
    s = d.soi_patch
    assert s.v_inf_km_s == pytest.approx(
        math.hypot(s.v_inf_radial_km_s, s.v_inf_tangential_km_s), rel=1e-12
    )
    assert s.v_inf_tangential_km_s == pytest.approx(
        s.v_sc_tangential_km_s - s.v_moon_km_s, rel=1e-12
    )
    assert s.v_inf_radial_km_s == pytest.approx(s.v_sc_radial_km_s, rel=1e-12)
    # 月球段必须从 SOI 补丁得到的 v_inf 出发，而不是地心二体延续
    assert d.moon_leg.v_inf_km_s == s.v_inf_km_s
    assert s.soi_radius_km == d.soi_patch.soi_radius_km


def test_total_tof_is_sum_of_two_legs():
    d = _designer().from_c3(-1.5)
    assert d.total_tof_days == pytest.approx(
        d.earth_leg.tof_to_soi_days + d.moon_leg.tof_soi_to_perilune_days, rel=1e-12
    )
    assert d.earth_leg.tof_to_soi_days > 0 and d.moon_leg.tof_soi_to_perilune_days > 0


def test_capture_dv_positive_and_below_hyperbolic_speed():
    d = _designer().from_c3(-1.5)
    m = d.moon_leg
    assert 0.0 < m.capture_dv_km_s < m.perilune_speed_hyperbolic_km_s
    assert m.capture_orbit_perilune_speed_km_s < m.perilune_speed_hyperbolic_km_s


def test_circular_capture_costs_more_than_elliptical():
    ell = _designer().from_c3(-1.5)
    circ = EarthMoonTransferDesigner(ParkingOrbit(185.0), CaptureOrbit(110.0)).from_c3(-1.5)
    assert circ.moon_leg.capture_dv_km_s > ell.moon_leg.capture_dv_km_s


def test_unreachable_c3_raises():
    with pytest.raises(NoSolutionError):
        _designer().from_c3(_designer().min_c3_km2_s2 - 0.5)


def test_too_long_tof_raises():
    with pytest.raises(NoSolutionError):
        _designer().from_total_tof(30.0)


def test_invalid_inputs_raise():
    with pytest.raises(ValidationError):
        EarthMoonTransferDesigner(ParkingOrbit(-10.0), CaptureOrbit(110.0, 310.0))
    with pytest.raises(ValidationError):
        EarthMoonTransferDesigner(ParkingOrbit(185.0), CaptureOrbit(-5.0))
    with pytest.raises(ValidationError):
        EarthMoonTransferDesigner(ParkingOrbit(185.0), CaptureOrbit(310.0, 110.0))
    with pytest.raises(ValidationError):
        _designer().from_total_tof(0.0)


def test_moon_phase_angle_consistent_with_moon_motion():
    designer = _designer()
    d = designer.from_c3(-1.5)
    s = d.soi_patch
    tof_s = d.earth_leg.tof_to_soi_days * DAY_S
    arrival_lon = (s.moon_phase_at_departure_deg + math.degrees(designer.omega_moon_rad_s * tof_s)) % 360.0
    assert arrival_lon == pytest.approx(d.earth_leg.transfer_angle_deg % 360.0, abs=1e-6)
