import streamlit as st
from __future__ import annotations
import sys
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# 1. Energy cost of running as a function of slope (Minetti et al. 2002)
# ---------------------------------------------------------------------------
def cost_of_running(slope: float) -> float:
    """
    Mass-specific energy cost of running, in J/(kg*m), as a function of
    slope 'i' (decimal fraction, e.g. 0.10 = +10% uphill, -0.10 = -10% downhill).

    Minetti's polynomial fit, valid roughly for -0.45 <= i <= +0.45.
    Outside that range the slope is clamped, since the fit isn't
    empirically supported beyond it (real GPX segments this steep are
    almost always smoothing/noise artifacts anyway).
    """
    i = max(-0.45, min(0.45, slope))
    return (155.4 * i**5 - 30.4 * i**4 - 43.3 * i**3
            + 46.3 * i**2 + 19.5 * i + 3.6)


FLAT_COST = cost_of_running(0.0)  # 3.6 J/kg/m, by construction


# ---------------------------------------------------------------------------
# 2. GPX parsing (no external dependencies required)
# ---------------------------------------------------------------------------
@dataclass
class Point:
    lat: float
    lon: float
    ele: float  # meters


def parse_gpx(path: str) -> list[Point]:
    tree = ET.parse(path)
    root = tree.getroot()

    # Auto-detect the GPX namespace from the root tag
    ns_uri = root.tag[1:root.tag.index("}")] if root.tag.startswith("{") else ""
    ns = {"gpx": ns_uri} if ns_uri else {}

    def findall(tag):
        return root.findall(f".//gpx:{tag}", ns) if ns else root.findall(f".//{tag}")

    def find_child(el, tag):
        return el.find(f"gpx:{tag}", ns) if ns else el.find(tag)

    points = []
    for trkpt in findall("trkpt"):
        lat = float(trkpt.get("lat"))
        lon = float(trkpt.get("lon"))
        ele_el = find_child(trkpt, "ele")
        ele = float(ele_el.text) if ele_el is not None and ele_el.text else 0.0
        points.append(Point(lat, lon, ele))

    if not points:
        # Fall back to route/way points if there's no track
        for tag in ("rtept", "wpt"):
            for pt in findall(tag):
                lat = float(pt.get("lat"))
                lon = float(pt.get("lon"))
                ele_el = find_child(pt, "ele")
                ele = float(ele_el.text) if ele_el is not None and ele_el.text else 0.0
                points.append(Point(lat, lon, ele))

    return points


def haversine(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance between two lat/lon points, in meters."""
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# 3. Elevation smoothing (raw GPX elevation is noisy -> slope noise blows up cost)
# ---------------------------------------------------------------------------
def smooth_elevation(points: list[Point], window: int = 5) -> list[Point]:
    if window <= 1 or len(points) < window:
        return points
    eles = [p.ele for p in points]
    half = window // 2
    smoothed = []
    for idx in range(len(eles)):
        lo, hi = max(0, idx - half), min(len(eles), idx + half + 1)
        smoothed.append(sum(eles[lo:hi]) / (hi - lo))
    return [Point(p.lat, p.lon, s) for p, s in zip(points, smoothed)]


# ---------------------------------------------------------------------------
# 4. Core EFD calculation
# ---------------------------------------------------------------------------
@dataclass
class EFDResult:
    horizontal_distance_m: float
    d_plus_m: float
    d_minus_m: float
    total_energy_j_per_kg: float
    efd_m: float
    naive_efd_m: float  # "1000 m D+ = 10 km flat" rule, for comparison

    @property
    def horizontal_distance_km(self): return self.horizontal_distance_m / 1000
    @property
    def efd_km(self): return self.efd_m / 1000
    @property
    def naive_efd_km(self): return self.naive_efd_m / 1000

    @property
    def overestimation_pct(self):
        return 100 * (self.naive_efd_m - self.efd_m) / self.efd_m


def compute_efd(points: list[Point], min_segment_m: float = 3.0) -> EFDResult:
    """
    Walk the trace segment by segment, compute local slope, look up the
    Minetti cost per meter at that slope, and integrate total energy.

    Note: because EFD = total_energy / flat_cost, and energy is mass-specific
    (J per kg), body mass cancels out entirely — EFD doesn't depend on the
    runner's weight, only on the trace's shape.

    min_segment_m: segments shorter than this are accumulated into the next
    one before computing slope, since very short segments massively amplify
    GPS/elevation noise (a 1 m horizontal step with 0.5 m of elevation noise
    looks like a 50% grade).
    """
    total_horizontal = 0.0
    d_plus = 0.0
    d_minus = 0.0
    total_energy = 0.0  # J per kg of body mass

    pending_horiz = 0.0
    pending_vert = 0.0

    for p0, p1 in zip(points, points[1:]):
        dh = haversine(p0.lat, p0.lon, p1.lat, p1.lon)
        dv = p1.ele - p0.ele

        pending_horiz += dh
        pending_vert += dv

        if pending_horiz < min_segment_m:
            continue

        seg_horiz, seg_vert = pending_horiz, pending_vert
        pending_horiz = pending_vert = 0.0

        seg_dist3d = math.hypot(seg_horiz, seg_vert)
        if seg_dist3d == 0:
            continue

        slope = seg_vert / seg_horiz if seg_horiz > 0 else 0.0

        total_horizontal += seg_horiz
        if seg_vert > 0:
            d_plus += seg_vert
        else:
            d_minus += -seg_vert

        cost = cost_of_running(slope)      # J/(kg*m)
        total_energy += cost * seg_dist3d  # integrate over actual (3D) distance covered

    efd_m = total_energy / FLAT_COST
    naive_efd_m = total_horizontal + d_plus * 10  # "1000 m D+ = 10 km flat"

    return EFDResult(
        horizontal_distance_m=total_horizontal,
        d_plus_m=d_plus,
        d_minus_m=d_minus,
        total_energy_j_per_kg=total_energy,
        efd_m=efd_m,
        naive_efd_m=naive_efd_m,
    )


# ---------------------------------------------------------------------------
# 5. CLI
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python gpx_efd.py <trace.gpx> [smoothing_window]")
        sys.exit(1)

    gpx_path = sys.argv[1]
    window = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    points = parse_gpx(gpx_path)
    if len(points) < 2:
        print("Not enough track points found in GPX file.")
        sys.exit(1)

    points = smooth_elevation(points, window=window)
    result = compute_efd(points)

    print(f"File: {gpx_path}")
    print(f"Track points: {len(points)} (elevation smoothing window: {window})")
    print(f"Horizontal distance: {result.horizontal_distance_km:.2f} km")
    print(f"D+: {result.d_plus_m:.0f} m   D-: {result.d_minus_m:.0f} m")
    print()
    print(f"Bioenergetic Equivalent Flat Distance (EFD): {result.efd_km:.2f} km")
    print(f"Naive '1000 m D+ = 10 km' estimate:           {result.naive_efd_km:.2f} km")
    print(f"Naive rule overestimates EFD by: {result.overestimation_pct:+.1f}%")


if __name__ == "__main__":
    main()
