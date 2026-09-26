"""地月转移补丁圆锥（patched-conic）任务设计模块。

从地球停车轨道、月球 SOI、目标近月点和捕获轨道出发，计算出发 C3、
飞行时间与近月捕获需求。模型显式分为地球段与月球段，并在月球 SOI
处做坐标系补丁 —— 不把整程当成一条地心二体轨道：

1. 地球段（地心圆锥曲线）：航天器自停车轨道近地点出发，沿地心圆锥
   曲线（C3<0 为椭圆、C3>0 为双曲线）传播到月球 SOI 边界；
2. SOI 边界补丁：在进入点把航天器地心速度减去月球地心轨道速度，
   得到月心双曲线剩余速度 v_inf，位置切换到月心系 SOI 球面；
3. 月球段（月心双曲线）：以 |v_inf| 与目标近月点半径构造月心双曲线，
   自 SOI（月心距 R_SOI）传播到近月点，计算进入捕获轨道的脉冲。

假设：月球圆轨道、转移共面、脉冲机动；SOI 进入点取地月连线共线近似，
近月点瞄准所需的横向偏移（冲击参数 b）由 B 平面/出发渐近线选择吸收。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from orbitforge.core.constants import (
    DAY_S,
    MU_EARTH_KM3_S2,
    MU_MOON_KM3_S2,
    MOON_SEMIMAJOR_KM,
    R_EARTH_EQUATOR_KM,
    R_MOON_KM,
)
from orbitforge.core.errors import NoSolutionError, ValidationError

__all__ = [
    "ParkingOrbit",
    "CaptureOrbit",
    "EarthLegResult",
    "SoiPatchResult",
    "MoonLegResult",
    "EarthMoonTransferDesign",
    "EarthMoonTransferDesigner",
]

_CONIC_LABEL = {"ellipse": "椭圆", "parabola": "抛物线", "hyperbola": "双曲线"}


@dataclass(frozen=True)
class ParkingOrbit:
    """地球圆停车轨道（出发近地点）。"""
    altitude_km: float

    @property
    def radius_km(self) -> float:
        return R_EARTH_EQUATOR_KM + self.altitude_km


@dataclass(frozen=True)
class CaptureOrbit:
    """目标月心捕获轨道；apolune_alt_km 为 None 时圆化于近月点。"""
    perilune_alt_km: float
    apolune_alt_km: float | None = None

    @property
    def perilune_radius_km(self) -> float:
        return R_MOON_KM + self.perilune_alt_km

    @property
    def apolune_radius_km(self) -> float:
        alt = self.perilune_alt_km if self.apolune_alt_km is None else self.apolune_alt_km
        return R_MOON_KM + alt


@dataclass(frozen=True)
class _ConicPoint:
    """圆锥曲线上某半径处的状态（离港段，真近点角 0..pi）。"""
    r_km: float
    tof_s: float
    speed_km_s: float
    radial_km_s: float
    tangential_km_s: float
    true_anomaly_rad: float


class _PerigeeConic:
    """自近心点出发的圆锥曲线；能量以 C3 = 2*eps 表示（<0 椭圆、=0 抛物线、>0 双曲线）。"""

    def __init__(self, mu: float, r_periapsis_km: float, c3_km2_s2: float):
        self.mu = mu
        self.r_p = r_periapsis_km
        self.c3 = c3_km2_s2
        v_p2 = 2.0 * mu / r_periapsis_km + c3_km2_s2
        if v_p2 <= 0.0:
            raise NoSolutionError("C3 过低：近心点速度非实数")
        self.v_p = math.sqrt(v_p2)
        self.h = r_periapsis_km * self.v_p
        self.p = self.h * self.h / mu
        if abs(c3_km2_s2) < 1e-12:
            self.kind = "parabola"
            self.a = math.inf
            self.e = 1.0
        else:
            self.a = -mu / c3_km2_s2
            self.e = 1.0 - r_periapsis_km / self.a
            self.kind = "ellipse" if c3_km2_s2 < 0.0 else "hyperbola"

    def at_radius(self, r_km: float) -> _ConicPoint:
        v2 = 2.0 * self.mu / r_km + self.c3
        if v2 < 0.0:
            raise NoSolutionError("圆锥曲线能量不足，达不到该半径")
        v = math.sqrt(v2)
        v_t = self.h / r_km
        if v_t > v * (1.0 + 1e-9):
            raise NoSolutionError("目标半径超出圆锥曲线远地点")
        v_r = math.sqrt(max(v * v - v_t * v_t, 0.0))
        cos_nu = max(-1.0, min(1.0, (self.p / r_km - 1.0) / self.e))
        nu = math.acos(cos_nu)
        if self.kind == "ellipse":
            cos_e = max(-1.0, min(1.0, (1.0 - r_km / self.a) / self.e))
            E = math.acos(cos_e)
            tof = math.sqrt(self.a ** 3 / self.mu) * (E - self.e * math.sin(E))
        elif self.kind == "hyperbola":
            F = math.acosh(max(1.0, (1.0 - r_km / self.a) / self.e))
            tof = math.sqrt((-self.a) ** 3 / self.mu) * (self.e * math.sinh(F) - F)
        else:
            D = math.tan(nu / 2.0)
            tof = 0.5 * math.sqrt(self.p ** 3 / self.mu) * (D + D ** 3 / 3.0)
        return _ConicPoint(r_km, tof, v, v_r, v_t, nu)


@dataclass(frozen=True)
class EarthLegResult:
    """地球段：地心圆锥曲线，停车轨道近地点 -> 月球 SOI 边界。"""
    c3_km2_s2: float
    conic_kind: str
    departure_dv_km_s: float
    perigee_speed_km_s: float
    transfer_angle_deg: float
    tof_to_soi_days: float
    soi_entry_geocentric_radius_km: float
    soi_entry_speed_km_s: float
    soi_entry_flight_path_deg: float


@dataclass(frozen=True)
class SoiPatchResult:
    """SOI 边界补丁：地心系 -> 月心系的显式速度变换。"""
    soi_radius_km: float
    v_sc_radial_km_s: float
    v_sc_tangential_km_s: float
    v_moon_km_s: float
    v_inf_radial_km_s: float
    v_inf_tangential_km_s: float
    v_inf_km_s: float
    moon_phase_at_departure_deg: float


@dataclass(frozen=True)
class MoonLegResult:
    """月球段：月心双曲线，SOI -> 近月点，以及捕获需求。"""
    v_inf_km_s: float
    perilune_radius_km: float
    impact_parameter_km: float
    turn_angle_deg: float
    tof_soi_to_perilune_days: float
    perilune_speed_hyperbolic_km_s: float
    capture_orbit_perilune_speed_km_s: float
    capture_dv_km_s: float
    capture_period_h: float


@dataclass(frozen=True)
class EarthMoonTransferDesign:
    parking: ParkingOrbit
    capture: CaptureOrbit
    earth_leg: EarthLegResult
    soi_patch: SoiPatchResult
    moon_leg: MoonLegResult
    total_dv_km_s: float
    total_tof_days: float

    def summary(self) -> str:
        e, s, m = self.earth_leg, self.soi_patch, self.moon_leg
        cap = self.capture
        apo = cap.perilune_alt_km if cap.apolune_alt_km is None else cap.apolune_alt_km
        lines = [
            "========== 地月转移任务设计（补丁圆锥，两段显式模型） ==========",
            "[输入]",
            f"  地球停车轨道高度  {self.parking.altitude_km:10.1f} km   (r = {self.parking.radius_km:.1f} km)",
            f"  目标近月点高度    {cap.perilune_alt_km:10.1f} km   (r = {cap.perilune_radius_km:.1f} km)",
            f"  捕获轨道          {cap.perilune_alt_km:6.1f} x {apo:.1f} km",
            "[地球段] 地心圆锥曲线：近地点 -> 月球 SOI 边界",
            f"  出发 C3           {e.c3_km2_s2:10.4f} km^2/s^2  ({_CONIC_LABEL[e.conic_kind]})",
            f"  近地点速度        {e.perigee_speed_km_s:10.4f} km/s",
            f"  出发脉冲 dV       {e.departure_dv_km_s:10.4f} km/s",
            f"  地心转移角        {e.transfer_angle_deg:10.2f} deg",
            f"  近地点 -> SOI     {e.tof_to_soi_days:10.3f} d",
            "[SOI 边界补丁] 地心系 -> 月心系",
            f"  SOI 半径          {s.soi_radius_km:10.1f} km   (进入点地心距 {e.soi_entry_geocentric_radius_km:.1f} km)",
            f"  航天器地心速度    {e.soi_entry_speed_km_s:10.4f} km/s  (径向 {s.v_sc_radial_km_s:+.4f} / 切向 {s.v_sc_tangential_km_s:+.4f})",
            f"  月球轨道速度      {s.v_moon_km_s:10.4f} km/s",
            f"  月心 v_inf        {s.v_inf_km_s:10.4f} km/s  (径向 {s.v_inf_radial_km_s:+.4f} / 切向 {s.v_inf_tangential_km_s:+.4f})",
            f"  出发时月相角      {s.moon_phase_at_departure_deg:10.2f} deg",
            "[月球段] 月心双曲线：SOI -> 近月点",
            f"  近月点双曲线速度  {m.perilune_speed_hyperbolic_km_s:10.4f} km/s",
            f"  瞄准参数 b        {m.impact_parameter_km:10.1f} km",
            f"  转向角            {m.turn_angle_deg:10.2f} deg",
            f"  SOI -> 近月点     {m.tof_soi_to_perilune_days:10.3f} d",
            "[捕获] 近月点脉冲",
            f"  捕获轨道近月点速度 {m.capture_orbit_perilune_speed_km_s:9.4f} km/s",
            f"  捕获 dV           {m.capture_dv_km_s:10.4f} km/s",
            f"  捕获轨道周期      {m.capture_period_h:10.2f} h",
            "[总计]",
            f"  总 dV             {self.total_dv_km_s:10.4f} km/s",
            f"  总飞行时间        {self.total_tof_days:10.3f} d",
        ]
        return "\n".join(lines)


class EarthMoonTransferDesigner:
    """地月转移设计器：给定停车轨道与捕获轨道，按 C3 或总飞行时间求解。"""

    def __init__(
        self,
        parking: ParkingOrbit,
        capture: CaptureOrbit,
        moon_distance_km: float = MOON_SEMIMAJOR_KM,
    ):
        if parking.altitude_km <= 0.0:
            raise ValidationError("停车轨道高度必须为正")
        if capture.perilune_alt_km <= 0.0:
            raise ValidationError("近月点高度必须为正（月面之上）")
        if capture.apolune_alt_km is not None and capture.apolune_alt_km < capture.perilune_alt_km:
            raise ValidationError("捕获轨道远月点不得低于近月点")
        if moon_distance_km <= 0.0:
            raise ValidationError("地月距离必须为正")
        self.parking = parking
        self.capture = capture
        self.moon_distance_km = moon_distance_km
        # 月球 Laplace SOI：地球段与月球段的显式边界
        self.soi_radius_km = moon_distance_km * (MU_MOON_KM3_S2 / MU_EARTH_KM3_S2) ** 0.4
        self.v_moon_km_s = math.sqrt(MU_EARTH_KM3_S2 / moon_distance_km)
        self.omega_moon_rad_s = self.v_moon_km_s / moon_distance_km
        # SOI 进入点（共线近似）：地月连线上靠地球一侧的 SOI 球面
        self.soi_entry_geocentric_km = moon_distance_km - self.soi_radius_km

    @property
    def min_c3_km2_s2(self) -> float:
        """到达 SOI 边界的最小 C3（远地点恰在 SOI 进入点的地心椭圆）。"""
        a = 0.5 * (self.parking.radius_km + self.soi_entry_geocentric_km)
        return -MU_EARTH_KM3_S2 / a

    def min_energy(self) -> EarthMoonTransferDesign:
        return self.from_c3(self.min_c3_km2_s2 + 1e-9)

    def from_c3(self, c3_km2_s2: float) -> EarthMoonTransferDesign:
        if c3_km2_s2 < self.min_c3_km2_s2:
            raise NoSolutionError(
                f"C3={c3_km2_s2:.4f} km^2/s^2 无法到达月球 SOI（最小 {self.min_c3_km2_s2:.4f}）"
            )
        # ---- 地球段：地心圆锥曲线，近地点 -> SOI 边界 ----
        earth = _PerigeeConic(MU_EARTH_KM3_S2, self.parking.radius_km, c3_km2_s2)
        dv_depart = earth.v_p - math.sqrt(MU_EARTH_KM3_S2 / self.parking.radius_km)
        entry = earth.at_radius(self.soi_entry_geocentric_km)

        # ---- SOI 边界补丁：v_inf = v_sc(地心) - v_moon(地心) ----
        v_inf_r = entry.radial_km_s
        v_inf_t = entry.tangential_km_s - self.v_moon_km_s
        v_inf = math.hypot(v_inf_r, v_inf_t)
        phase_deg = math.degrees(
            entry.true_anomaly_rad - self.omega_moon_rad_s * entry.tof_s
        ) % 360.0

        # ---- 月球段：月心双曲线，SOI -> 近月点 ----
        r_perilune = self.capture.perilune_radius_km
        moon = _PerigeeConic(MU_MOON_KM3_S2, r_perilune, v_inf * v_inf)
        impact_km = -moon.a * math.sqrt(moon.e * moon.e - 1.0)
        turn_deg = math.degrees(2.0 * math.asin(1.0 / moon.e))
        t_moon_leg_s = moon.at_radius(self.soi_radius_km).tof_s

        # ---- 近月点捕获脉冲 ----
        r_apolune = self.capture.apolune_radius_km
        a_cap = 0.5 * (r_perilune + r_apolune)
        v_cap = math.sqrt(MU_MOON_KM3_S2 * (2.0 / r_perilune - 1.0 / a_cap))
        dv_capture = moon.v_p - v_cap
        period_h = 2.0 * math.pi * math.sqrt(a_cap ** 3 / MU_MOON_KM3_S2) / 3600.0

        earth_leg = EarthLegResult(
            c3_km2_s2=c3_km2_s2,
            conic_kind=earth.kind,
            departure_dv_km_s=dv_depart,
            perigee_speed_km_s=earth.v_p,
            transfer_angle_deg=math.degrees(entry.true_anomaly_rad),
            tof_to_soi_days=entry.tof_s / DAY_S,
            soi_entry_geocentric_radius_km=entry.r_km,
            soi_entry_speed_km_s=entry.speed_km_s,
            soi_entry_flight_path_deg=math.degrees(math.atan2(entry.radial_km_s, entry.tangential_km_s)),
        )
        patch = SoiPatchResult(
            soi_radius_km=self.soi_radius_km,
            v_sc_radial_km_s=entry.radial_km_s,
            v_sc_tangential_km_s=entry.tangential_km_s,
            v_moon_km_s=self.v_moon_km_s,
            v_inf_radial_km_s=v_inf_r,
            v_inf_tangential_km_s=v_inf_t,
            v_inf_km_s=v_inf,
            moon_phase_at_departure_deg=phase_deg,
        )
        moon_leg = MoonLegResult(
            v_inf_km_s=v_inf,
            perilune_radius_km=r_perilune,
            impact_parameter_km=impact_km,
            turn_angle_deg=turn_deg,
            tof_soi_to_perilune_days=t_moon_leg_s / DAY_S,
            perilune_speed_hyperbolic_km_s=moon.v_p,
            capture_orbit_perilune_speed_km_s=v_cap,
            capture_dv_km_s=dv_capture,
            capture_period_h=period_h,
        )
        return EarthMoonTransferDesign(
            parking=self.parking,
            capture=self.capture,
            earth_leg=earth_leg,
            soi_patch=patch,
            moon_leg=moon_leg,
            total_dv_km_s=dv_depart + dv_capture,
            total_tof_days=(entry.tof_s + t_moon_leg_s) / DAY_S,
        )

    def from_total_tof(self, tof_days: float) -> EarthMoonTransferDesign:
        """按总飞行时间（近地点 -> 近月点）反解出发 C3。"""
        if tof_days <= 0.0:
            raise ValidationError("飞行时间必须为正")

        def residual(c3: float) -> float:
            return self.from_c3(c3).total_tof_days - tof_days

        lo = self.min_c3_km2_s2 + 1e-9
        if residual(lo) < 0.0:
            raise NoSolutionError("所需飞行时间长于最小能量转移，无法达到")
        hi = 10.0
        while residual(hi) > 0.0:
            hi *= 2.0
            if hi > 1e4:
                raise NoSolutionError("所需飞行时间过短，超出合理 C3 范围")
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if residual(mid) > 0.0:
                lo = mid
            else:
                hi = mid
        return self.from_c3(0.5 * (lo + hi))


if __name__ == "__main__":
    designer = EarthMoonTransferDesigner(
        parking=ParkingOrbit(altitude_km=185.0),
        capture=CaptureOrbit(perilune_alt_km=110.0, apolune_alt_km=310.0),
    )
    print(designer.min_energy().summary())
    print()
    print(designer.from_total_tof(3.0).summary())
    print()
    print(designer.from_c3(-1.5).summary())
