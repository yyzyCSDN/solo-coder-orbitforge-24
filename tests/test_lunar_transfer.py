import math
from orbitforge.core.constants import DAY_S, MU_MOON_KM3_S2, R_MOON_KM
from orbitforge.interplanetary.lunar_transfer import (
    CaptureOrbit, ParkingOrbit, design_lunar_transfer, hohmann_transfer_to_moon,
    max_capturable_v_infinity_km_s, moon_soi_radius_km, patch_at_moon_soi,
    selenocentric_approach, transfer_from_c3,
)


def _baseline():
    return design_lunar_transfer(ParkingOrbit.from_altitude(200.0),
                                 CaptureOrbit.circular(100.0))


def test_hohmann_earth_segment_c3_tof_dv():
    d = _baseline()
    assert abs(d.transfer.c3_km2_s2 - (-2.03)) < 0.05
    assert abs(d.departure_delta_v_km_s - 3.13) < 0.02
    assert abs(d.transfer.time_of_flight_s / DAY_S - 4.98) < 0.05
    assert not d.transfer.is_hyperbolic


def test_soi_boundary_is_explicit_frame_transformation():
    d = _baseline()
    s = d.soi_crossing
    # vector subtraction at the boundary, not a scalar speed difference
    assert abs(s.v_infinity_km_s
               - math.hypot(s.v_infinity_radial_km_s,
                            s.v_infinity_transverse_km_s)) < 1e-12
    arr = d.transfer.arrival
    assert abs(s.v_infinity_transverse_km_s
               - (arr.transverse_speed_km_s - s.moon_orbital_speed_km_s)) < 1e-12
    assert abs(s.v_infinity_km_s - 0.84) < 0.02
    assert abs(s.soi_radius_km - 66200) < 300


def test_moon_segment_uses_lunar_mu_and_capture_requirements():
    d = _baseline()
    ap = d.approach
    rp = d.capture_orbit.perilune_radius_km
    assert abs(ap.eccentricity
               - (1.0 + rp * ap.v_infinity_km_s ** 2 / MU_MOON_KM3_S2)) < 1e-12
    assert abs(d.capture_burn.delta_v_km_s - 0.82) < 0.02
    assert abs(ap.miss_distance_km - 5430) < 100
    assert abs(ap.soi_to_perilune_s / 3600.0 - 17.7) < 1.0


def test_phase_angle_and_totals():
    d = _baseline()
    assert abs(math.degrees(d.moon_phase_angle_rad) - 114.7) < 1.0
    assert abs(d.total_time_s / DAY_S - 5.7) < 0.2
    assert abs(d.total_delta_v_km_s
               - (d.departure_delta_v_km_s + d.capture_burn.delta_v_km_s)) < 1e-12


def test_fast_transfer_positive_c3():
    d = design_lunar_transfer(ParkingOrbit.from_altitude(200.0),
                              CaptureOrbit.circular(100.0), c3_km2_s2=5.0)
    assert d.transfer.is_hyperbolic
    assert d.transfer.time_of_flight_s / DAY_S < 1.5
    assert abs(d.soi_crossing.v_infinity_km_s - 2.78) < 0.05
    assert abs(d.capture_burn.delta_v_km_s - 1.98) < 0.05
    # faster transfer costs more total delta-v than Hohmann
    assert d.total_delta_v_km_s > _baseline().total_delta_v_km_s


def test_parabolic_limit_c3_zero():
    d = design_lunar_transfer(ParkingOrbit.from_altitude(200.0),
                              CaptureOrbit.circular(100.0), c3_km2_s2=0.0)
    tof_d = d.transfer.time_of_flight_s / DAY_S
    assert 2.0 < tof_d < 2.2


def test_elliptical_capture_orbit():
    d = design_lunar_transfer(ParkingOrbit.from_altitude(200.0),
                              CaptureOrbit.from_altitudes(100.0, 10000.0))
    # less delta-v than capturing directly into the 100 km circular orbit
    assert 0.3 < d.capture_burn.delta_v_km_s < _baseline().capture_burn.delta_v_km_s


def test_max_capturable_v_infinity():
    orb = CaptureOrbit.circular(100.0)
    v_inf = max_capturable_v_infinity_km_s(orb, 1.0)
    ap = selenocentric_approach(v_inf, orb.perilune_radius_km,
                                moon_soi_radius_km())
    burn = ap.perilune_speed_km_s - math.sqrt(
        MU_MOON_KM3_S2 / orb.perilune_radius_km)
    assert abs(burn - 1.0) < 1e-9


def test_unreachable_transfer_raises():
    try:
        transfer_from_c3(6578.137, -10.0)
        assert False
    except ValueError:
        pass


def test_departure_slower_than_circular_raises():
    try:
        design_lunar_transfer(ParkingOrbit.from_altitude(200.0),
                              CaptureOrbit.circular(100.0), c3_km2_s2=-100.0)
        assert False
    except ValueError:
        pass


def test_invalid_orbits_raise():
    for bad in (lambda: ParkingOrbit.from_altitude(-10.0),
                lambda: CaptureOrbit.circular(-10.0),
                lambda: CaptureOrbit.from_altitudes(500.0, 100.0),
                lambda: transfer_from_c3(6578.137, -200.0)):
        try:
            bad()
            assert False
        except ValueError:
            pass


def test_hohmann_helper_matches_design_default():
    d = _baseline()
    t = hohmann_transfer_to_moon(6578.137)
    assert abs(d.transfer.c3_km2_s2 - t.c3_km2_s2) < 1e-9
    assert abs(d.transfer.time_of_flight_s - t.time_of_flight_s) < 1e-6


def test_summary_renders_all_segments():
    text = _baseline().summary()
    assert 'Earth' in text and 'SOI' in text and 'Moon' in text
    assert 'C3' in text and 'capture delta-v' in text
