"""Earth-Moon patched-conic mission design.

A translunar trajectory is assembled from three explicitly separated
pieces and is never propagated as a single Earth two-body orbit:

1. Earth segment (geocentric two-body): circular parking orbit, TLI burn
   onto a geocentric transfer conic (ellipse, parabola or hyperbola) out
   to the Moon's orbit.
2. Moon SOI boundary: Galilean transformation from the geocentric
   inertial frame into the selenocentric frame, yielding the arrival
   hyperbolic excess velocity v_inf.
3. Moon segment (selenocentric two-body): hyperbolic approach to the
   perilune and the capture burn into the target lunar orbit.

Each leg uses its own central-body gravitational parameter and the legs
are patched at the lunar sphere of influence (SOI).

Assumptions: coplanar geometry, circular lunar orbit at ``a_moon_km``,
impulsive burns, spherical bodies. The geocentric leg is evaluated at the
Moon's orbital radius; the SOI radius (~0.17 lunar distances) sets the
scale of this standard patched-conic approximation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from orbitforge.core.constants import (
    DAY_S,
    MU_EARTH_KM3_S2,
    MU_MOON_KM3_S2,
    MOON_SMA_KM,
    RAD2DEG,
    R_EARTH_EQUATOR_KM,
    R_MOON_KM,
)

_TWO_PI = 2.0 * math.pi
_PARABOLIC_C3_TOL_KM2_S2 = 1e-9


def moon_soi_radius_km(a_moon_km: float = MOON_SMA_KM) -> float:
    """Laplace sphere-of-influence radius of the Moon [km]."""
    return a_moon_km * (MU_MOON_KM3_S2 / MU_EARTH_KM3_S2) ** (2.0 / 5.0)


def moon_orbital_speed_km_s(a_moon_km: float = MOON_SMA_KM) -> float:
    """Moon's speed relative to Earth on its (assumed circular) orbit."""
    return math.sqrt((MU_EARTH_KM3_S2 + MU_MOON_KM3_S2) / a_moon_km)


def _clamp(x: float) -> float:
    return max(-1.0, min(1.0, x))


@dataclass(frozen=True)
class ConicState:
    """State on a conic at a given radius, outbound leg from periapsis."""
    radius_km: float
    speed_km_s: float
    radial_speed_km_s: float
    transverse_speed_km_s: float
    true_anomaly_rad: float
    time_since_periapsis_s: float

    @property
    def flight_path_angle_rad(self) -> float:
        return math.atan2(self.radial_speed_km_s, self.transverse_speed_km_s)


def _conic_state_at_radius(energy_km2_s2: float, h_km2_s: float, mu_km3_s2: float,
                           radius_km: float) -> tuple[ConicState, float]:
    """Conic state at ``radius_km`` for any energy (ellipse/parabola/hyperbola).

    Returns (state, eccentricity). Raises ValueError if the radius is not
    reached by the outbound leg.
    """
    mu = mu_km3_s2
    e = math.sqrt(max(0.0, 1.0 + 2.0 * energy_km2_s2 * h_km2_s ** 2 / mu ** 2))
    p = h_km2_s ** 2 / mu
    v_sq = 2.0 * (energy_km2_s2 + mu / radius_km)
    if v_sq <= 0.0:
        raise ValueError('radius not reached by this conic')
    speed = math.sqrt(v_sq)
    v_t = h_km2_s / radius_km
    if v_t > speed * (1.0 + 1e-12):
        raise ValueError('radius beyond apoapsis of this conic')
    v_r = math.sqrt(max(0.0, v_sq - v_t * v_t))
    nu = math.acos(_clamp((p / radius_km - 1.0) / e)) if e > 0.0 else 0.0
    c3 = 2.0 * energy_km2_s2
    if abs(c3) < _PARABOLIC_C3_TOL_KM2_S2:
        # Barker's equation
        d = math.tan(nu / 2.0)
        t = 0.5 * math.sqrt(p ** 3 / mu) * (d + d ** 3 / 3.0)
    elif energy_km2_s2 < 0.0:
        a = -mu / (2.0 * energy_km2_s2)
        big_e = math.acos(_clamp((1.0 - radius_km / a) / e))
        t = math.sqrt(a ** 3 / mu) * (big_e - e * math.sin(big_e))
    else:
        a = -mu / (2.0 * energy_km2_s2)  # negative semi-major axis
        big_h = math.acosh((1.0 - radius_km / a) / e)
        t = math.sqrt((-a) ** 3 / mu) * (e * math.sinh(big_h) - big_h)
    state = ConicState(radius_km, speed, v_r, v_t, nu, t)
    return state, e


@dataclass(frozen=True)
class ParkingOrbit:
    """Circular Earth parking orbit (TLI departure point)."""
    radius_km: float

    def __post_init__(self):
        if self.radius_km <= R_EARTH_EQUATOR_KM:
            raise ValueError('parking orbit radius must exceed Earth radius')

    @classmethod
    def from_altitude(cls, altitude_km: float) -> ParkingOrbit:
        return cls(R_EARTH_EQUATOR_KM + altitude_km)

    @property
    def altitude_km(self) -> float:
        return self.radius_km - R_EARTH_EQUATOR_KM

    @property
    def circular_speed_km_s(self) -> float:
        return math.sqrt(MU_EARTH_KM3_S2 / self.radius_km)


@dataclass(frozen=True)
class CaptureOrbit:
    """Target selenocentric orbit after the capture burn."""
    perilune_radius_km: float
    apolune_radius_km: float

    def __post_init__(self):
        if self.perilune_radius_km <= R_MOON_KM:
            raise ValueError('perilune radius must exceed lunar radius')
        if self.apolune_radius_km < self.perilune_radius_km:
            raise ValueError('apolune radius must be >= perilune radius')

    @classmethod
    def circular(cls, altitude_km: float) -> CaptureOrbit:
        r = R_MOON_KM + altitude_km
        return cls(r, r)

    @classmethod
    def from_altitudes(cls, perilune_altitude_km: float,
                       apolune_altitude_km: float | None = None) -> CaptureOrbit:
        r_p = R_MOON_KM + perilune_altitude_km
        r_a = R_MOON_KM + (perilune_altitude_km if apolune_altitude_km is None
                           else apolune_altitude_km)
        return cls(r_p, r_a)

    @property
    def semi_major_axis_km(self) -> float:
        return 0.5 * (self.perilune_radius_km + self.apolune_radius_km)

    @property
    def perilune_altitude_km(self) -> float:
        return self.perilune_radius_km - R_MOON_KM

    @property
    def apolune_altitude_km(self) -> float:
        return self.apolune_radius_km - R_MOON_KM


@dataclass(frozen=True)
class GeocentricTransfer:
    """Earth segment: geocentric conic from parking-orbit perigee (TLI) to
    the Moon's orbit. C3 is negative for the usual elliptical transfer."""
    perigee_radius_km: float
    arrival_radius_km: float
    semi_major_axis_km: float  # negative for hyperbolic departure, inf parabolic
    eccentricity: float
    c3_km2_s2: float
    perigee_speed_km_s: float
    arrival: ConicState

    @property
    def time_of_flight_s(self) -> float:
        return self.arrival.time_since_periapsis_s

    @property
    def is_hyperbolic(self) -> bool:
        return self.c3_km2_s2 > _PARABOLIC_C3_TOL_KM2_S2


def transfer_from_c3(parking_radius_km: float, c3_km2_s2: float,
                     arrival_radius_km: float = MOON_SMA_KM) -> GeocentricTransfer:
    """Geocentric transfer conic departing the parking-orbit perigee with the
    given C3, evaluated at ``arrival_radius_km`` (default: Moon's orbit).

    Pass ``arrival_radius_km = MOON_SMA_KM - moon_soi_radius_km()`` to evaluate
    the geocentric leg at SOI entry instead of the lunar-distance crossing.
    """
    v_p_sq = c3_km2_s2 + 2.0 * MU_EARTH_KM3_S2 / parking_radius_km
    if v_p_sq <= 0.0:
        raise ValueError('C3 too low: perigee speed is not real')
    v_p = math.sqrt(v_p_sq)
    energy = 0.5 * c3_km2_s2
    h = parking_radius_km * v_p
    if energy < 0.0:
        a = -MU_EARTH_KM3_S2 / (2.0 * energy)
        e = math.sqrt(max(0.0, 1.0 + 2.0 * energy * h * h / MU_EARTH_KM3_S2 ** 2))
        r_apo = a * (1.0 + e)
        if arrival_radius_km > r_apo * (1.0 + 1e-12):
            raise ValueError(
                f'transfer ellipse apogee {r_apo:.0f} km does not reach '
                f'{arrival_radius_km:.0f} km')
    arrival, e = _conic_state_at_radius(energy, h, MU_EARTH_KM3_S2, arrival_radius_km)
    a = -MU_EARTH_KM3_S2 / (2.0 * energy) if abs(energy) > 0.0 else math.inf
    return GeocentricTransfer(parking_radius_km, arrival_radius_km, a, e,
                              c3_km2_s2, v_p, arrival)


def hohmann_transfer_to_moon(parking_radius_km: float,
                             arrival_radius_km: float = MOON_SMA_KM) -> GeocentricTransfer:
    """Minimum-energy (Hohmann) geocentric transfer to the Moon's orbit."""
    a_t = 0.5 * (parking_radius_km + arrival_radius_km)
    v_p = math.sqrt(MU_EARTH_KM3_S2 * (2.0 / parking_radius_km - 1.0 / a_t))
    c3 = v_p * v_p - 2.0 * MU_EARTH_KM3_S2 / parking_radius_km
    return transfer_from_c3(parking_radius_km, c3, arrival_radius_km)


@dataclass(frozen=True)
class SoiCrossing:
    """Boundary between the Earth and Moon segments: geocentric arrival
    velocity transformed into the selenocentric frame (arrival v_inf)."""
    soi_radius_km: float
    moon_orbital_speed_km_s: float
    v_infinity_km_s: float
    v_infinity_radial_km_s: float
    v_infinity_transverse_km_s: float


def patch_at_moon_soi(transfer: GeocentricTransfer,
                      a_moon_km: float = MOON_SMA_KM) -> SoiCrossing:
    """Patch the geocentric leg onto the selenocentric leg at the Moon's SOI.

    v_inf = v_spacecraft(geocentric) - v_Moon(geocentric), component-wise in
    the radial/transverse basis at the lunar-orbit crossing. This Galilean
    transformation is the boundary the two segments are stitched at; the
    whole trajectory is never a single Earth two-body orbit.
    """
    v_moon = moon_orbital_speed_km_s(a_moon_km)
    v_inf_r = transfer.arrival.radial_speed_km_s
    v_inf_t = transfer.arrival.transverse_speed_km_s - v_moon
    return SoiCrossing(
        soi_radius_km=moon_soi_radius_km(a_moon_km),
        moon_orbital_speed_km_s=v_moon,
        v_infinity_km_s=math.hypot(v_inf_r, v_inf_t),
        v_infinity_radial_km_s=v_inf_r,
        v_infinity_transverse_km_s=v_inf_t,
    )


@dataclass(frozen=True)
class LunarApproach:
    """Moon segment: selenocentric hyperbola from SOI entry to perilune."""
    v_infinity_km_s: float
    perilune_radius_km: float
    eccentricity: float
    miss_distance_km: float       # asymptote offset (B-plane aim) from Moon center
    turn_angle_rad: float
    perilune_speed_km_s: float    # hyperbolic speed at perilune (pre-capture)
    soi_to_perilune_s: float      # time from SOI entry to perilune


def selenocentric_approach(v_infinity_km_s: float, perilune_radius_km: float,
                           soi_radius_km: float) -> LunarApproach:
    """Selenocentric approach hyperbola for the given arrival v_inf and
    target perilune radius (Moon-segment two-body dynamics, mu = MU_MOON)."""
    if v_infinity_km_s <= 0.0:
        raise ValueError('arrival v_inf must be positive')
    if perilune_radius_km <= R_MOON_KM:
        raise ValueError('perilune radius must exceed lunar radius')
    mu = MU_MOON_KM3_S2
    e = 1.0 + perilune_radius_km * v_infinity_km_s ** 2 / mu
    miss = perilune_radius_km * math.sqrt(
        1.0 + 2.0 * mu / (perilune_radius_km * v_infinity_km_s ** 2))
    turn = 2.0 * math.asin(1.0 / e)
    v_perilune = math.sqrt(v_infinity_km_s ** 2 + 2.0 * mu / perilune_radius_km)
    a = -mu / v_infinity_km_s ** 2
    big_h = math.acosh((1.0 - soi_radius_km / a) / e)
    t_soi = math.sqrt((-a) ** 3 / mu) * (e * math.sinh(big_h) - big_h)
    return LunarApproach(v_infinity_km_s, perilune_radius_km, e, miss, turn,
                         v_perilune, t_soi)


@dataclass(frozen=True)
class CaptureBurn:
    """Impulsive capture (LOI) burn at perilune."""
    arrival_speed_km_s: float
    orbit_speed_km_s: float
    delta_v_km_s: float


def capture_burn_at_perilune(approach: LunarApproach,
                             orbit: CaptureOrbit) -> CaptureBurn:
    """Retro-burn at perilune from the approach hyperbola into ``orbit``."""
    a = orbit.semi_major_axis_km
    v_target = math.sqrt(
        MU_MOON_KM3_S2 * (2.0 / orbit.perilune_radius_km - 1.0 / a))
    return CaptureBurn(approach.perilune_speed_km_s, v_target,
                       approach.perilune_speed_km_s - v_target)


def max_capturable_v_infinity_km_s(orbit: CaptureOrbit,
                                   delta_v_budget_km_s: float) -> float:
    """Largest arrival v_inf that ``delta_v_budget_km_s`` can still capture
    into ``orbit`` at its perilune."""
    a = orbit.semi_major_axis_km
    v_target = math.sqrt(
        MU_MOON_KM3_S2 * (2.0 / orbit.perilune_radius_km - 1.0 / a))
    v_hyp = v_target + delta_v_budget_km_s
    v_inf_sq = v_hyp * v_hyp - 2.0 * MU_MOON_KM3_S2 / orbit.perilune_radius_km
    return math.sqrt(max(0.0, v_inf_sq))


@dataclass(frozen=True)
class LunarTransferDesign:
    """Complete patched-conic Earth-Moon transfer design."""
    parking: ParkingOrbit
    capture_orbit: CaptureOrbit
    transfer: GeocentricTransfer
    departure_delta_v_km_s: float
    soi_crossing: SoiCrossing
    approach: LunarApproach
    capture_burn: CaptureBurn
    moon_phase_angle_rad: float  # Moon's lead angle ahead of the TLI point
    total_time_s: float          # TLI -> perilune, incl. SOI transit

    @property
    def total_delta_v_km_s(self) -> float:
        return self.departure_delta_v_km_s + self.capture_burn.delta_v_km_s

    def summary(self) -> str:
        t = self.transfer
        s = self.soi_crossing
        ap = self.approach
        cap = self.capture_burn
        conic = ('hyperbola' if t.is_hyperbolic else 'ellipse')
        lines = [
            '=' * 62,
            'Earth-Moon patched-conic mission design',
            '=' * 62,
            '[Segment 1 - Earth (geocentric two-body)]',
            f'  parking orbit altitude      {self.parking.altitude_km:12.1f} km',
            f'  TLI perigee speed           {t.perigee_speed_km_s:12.3f} km/s',
            f'  departure C3                {t.c3_km2_s2:12.2f} km^2/s^2',
            f'  departure delta-v (TLI)     {self.departure_delta_v_km_s:12.3f} km/s',
            f'  transfer conic              {conic:>12s}  a={t.semi_major_axis_km:.0f} km'
            f'  e={t.eccentricity:.4f}',
            f'  geocentric time of flight   {t.time_of_flight_s / DAY_S:12.2f} d',
            '[Boundary - Moon SOI patch (geocentric -> selenocentric)]',
            f'  SOI radius                  {s.soi_radius_km:12.0f} km',
            f'  Moon orbital speed          {s.moon_orbital_speed_km_s:12.3f} km/s',
            f'  arrival v_inf (selenoc.)    {s.v_infinity_km_s:12.3f} km/s',
            '[Segment 2 - Moon (selenocentric two-body)]',
            f'  perilune altitude           {self.capture_orbit.perilune_altitude_km:12.1f} km',
            f'  asymptote miss distance     {ap.miss_distance_km:12.0f} km',
            f'  hyperbolic perilune speed   {ap.perilune_speed_km_s:12.3f} km/s',
            f'  capture delta-v (LOI)       {cap.delta_v_km_s:12.3f} km/s',
            f'  capture orbit               {self.capture_orbit.perilune_altitude_km:9.0f} x'
            f' {self.capture_orbit.apolune_altitude_km:.0f} km alt',
            f'  time inside SOI             {ap.soi_to_perilune_s / 3600.0:12.1f} h',
            '[Totals]',
            f'  total delta-v               {self.total_delta_v_km_s:12.3f} km/s',
            f'  total time TLI -> perilune  {self.total_time_s / DAY_S:12.2f} d',
            f'  Moon phase angle at TLI     {self.moon_phase_angle_rad * RAD2DEG:12.1f} deg',
            '=' * 62,
        ]
        return '\n'.join(lines)


def design_lunar_transfer(parking: ParkingOrbit, capture_orbit: CaptureOrbit, *,
                          c3_km2_s2: float | None = None,
                          transfer: GeocentricTransfer | None = None,
                          a_moon_km: float = MOON_SMA_KM) -> LunarTransferDesign:
    """Design an Earth-Moon transfer by patched conics.

    Defaults to a Hohmann geocentric transfer; pass ``c3_km2_s2`` for a
    faster (or slower) direct transfer, or a fully custom ``transfer``.
    The Earth segment, the SOI boundary patch and the Moon segment are
    computed as separate two-body legs stitched at the lunar SOI.
    """
    # Segment 1: Earth (geocentric two-body).
    if transfer is None:
        if c3_km2_s2 is None:
            transfer = hohmann_transfer_to_moon(parking.radius_km, a_moon_km)
        else:
            transfer = transfer_from_c3(parking.radius_km, c3_km2_s2, a_moon_km)
    dep_dv = transfer.perigee_speed_km_s - parking.circular_speed_km_s
    if dep_dv <= 0.0:
        raise ValueError('transfer perigee speed is below parking-orbit '
                         'circular speed; not a departure')
    # Boundary: patch at the Moon's SOI (frame transformation).
    crossing = patch_at_moon_soi(transfer, a_moon_km)
    # Segment 2: Moon (selenocentric two-body).
    approach = selenocentric_approach(crossing.v_infinity_km_s,
                                      capture_orbit.perilune_radius_km,
                                      crossing.soi_radius_km)
    burn = capture_burn_at_perilune(approach, capture_orbit)
    # Phasing: Moon's lead angle ahead of the TLI departure point.
    n_moon = math.sqrt((MU_EARTH_KM3_S2 + MU_MOON_KM3_S2) / a_moon_km ** 3)
    phase = (transfer.arrival.true_anomaly_rad
             - n_moon * transfer.time_of_flight_s) % _TWO_PI
    total = transfer.time_of_flight_s + approach.soi_to_perilune_s
    return LunarTransferDesign(parking, capture_orbit, transfer, dep_dv,
                               crossing, approach, burn, phase, total)
