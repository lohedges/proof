from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import lsm


@dataclass
class Connector:
    """
    A short connective segments between two other segments.
    """
    a: lsm.LineSegment
    b: lsm.LineSegment
    connector: lsm.LineSegment


class PolyLine:
    """
    A set of line segments, joined together into a linear path
    """
    def __init__(self, l: lsm.LineSegment):
        self.points = [l.p1, l.p2]

    def append(self, other: lsm.LineSegment):
        """
        Add a line segment to the poly-line.
        It will automatically attach it in the correct place.
        """
        if other.p1 == self.points[0]:
            self.points.insert(0, other.p2)
        elif other.p1 == self.points[-1]:
            self.points.append(other.p2)
        elif other.p2 == self.points[0]:
            self.points.insert(0, other.p1)
        elif other.p2 == self.points[-1]:
            self.points.append(other.p1)
        else:
            raise ValueError(f"Line segment to add ({other}) does not touch this poly line ({self.points})")

    @property
    def length(self) -> float:
        length = 0
        for p1, p2 in zip(self.points, self.points[1:]):
            length += lsm.LineSegment(p1.x, p1.y, p2.x, p2.y).length
        return length


def trace_filaments(lines: List[lsm.LineSegment]) -> List[PolyLine]:
    """
    Given a set of line segments, merge some together into a list of poly-lines.

    It tries to find ends of two segments which are near to each other and also attached to aligned lines.
    """
    connected_segments = find_connectors(lines)

    polys: List[PolyLine] = []

    # First deal with all the segments which have no connectors on them
    for line in lines:
        if line not in connected_segments:
            polys.append(PolyLine(line))

    # Find all the segments which only have connectors on one end.
    ends = [line for line, cs in connected_segments.items() if len(cs[0]) == 0 or len(cs[1]) == 0]

    for end in ends:
        # Start by initialising the PolyLine with the end
        poly = PolyLine(end)
        # Start the algorithm with the starter
        next_segment = end
        connector_candidates = connected_segments[next_segment][0] or connected_segments[next_segment][1] or []
        while True:
            if not connector_candidates:  # 0 connectors
                break
            elif len(connector_candidates) == 1:  # 1 connector
                chosen = connector_candidates[0]
            else:  # > 1 connectors
                chosen = max(connector_candidates, key=lambda x: x.connector.length)

            poly.append(chosen.connector)

            next_segment = chosen.b if next_segment == chosen.a else chosen.a
            poly.append(next_segment)
            connector_candidates = connected_segments[next_segment][1] if chosen in connected_segments[next_segment][0] else connected_segments[next_segment][0]

            try:
                ends.remove(next_segment)
            except ValueError:
                pass

        polys.append(poly)

    return polys


def find_connectors(lines: List[lsm.LineSegment]) -> Dict[lsm.LineSegment, Tuple[List[Connector], List[Connector]]]:
    """
    For every pair of lines, checks whether any of the pairs of ends could be connected together.

    For each segment, returns all the valid connectors for each end of the line.
    """
    # A mapping of line segments to connectors attached to them
    # Each value is a tuple containing connectors on (p1, p2)
    connected_segments: Dict[lsm.LineSegment, Tuple[List[Connector], List[Connector]]] = defaultdict(lambda: ([], []))

    for L_1 in lines:
        #print("L_1:", L_1)
        for L_n in lines:
            if L_n == L_1:
                continue
            #print("L_n:", L_n)

            relative_delta_dist = 2.0
            delta_dist = min(L_n.length, L_1.length) * relative_delta_dist
            gap_line = lsm.closest_end_points(L_1, L_n)
            smallest_dist = gap_line.length
            if smallest_dist > delta_dist:
                continue

            delta_angle = 0.3
            angle_difference = lsm.abs_angle_difference(L_1.angle, L_n.angle)
            if angle_difference > delta_angle:
                continue

            L_1_extreme = L_1.p1 if L_1.p2 in {gap_line.p1, gap_line.p2} else L_1.p2
            L_n_extreme = L_n.p1 if L_n.p2 in {gap_line.p1, gap_line.p2} else L_n.p2
            L_1_close = L_1.p1 if L_1.p2 == L_1_extreme else L_1.p2
            L_n_close = L_n.p1 if L_n.p2 == L_n_extreme else L_n.p2

            aligned_L_1 = lsm.LineSegment(L_1_extreme.x, L_1_extreme.y, L_1_close.x, L_1_close.y)
            aligned_gap = lsm.LineSegment(L_1_close.x, L_1_close.y, L_n_close.x, L_n_close.y)
            aligned_L_n = lsm.LineSegment(L_n_close.x, L_n_close.y, L_n_extreme.x, L_n_extreme.y)

            #print("aligned_gap:", aligned_gap)

            absolute_gap_size = 5
            # TODO clock algebra
            if (abs(aligned_L_1.angle - aligned_gap.angle) < delta_angle and
                abs(aligned_gap.angle - aligned_L_n.angle) < delta_angle) \
                    or (aligned_L_1.angle - aligned_L_n.angle) < delta_angle \
                    and aligned_gap.length < absolute_gap_size:
                con = Connector(L_1, L_n, aligned_gap)
                connected_segments[L_1][int(L_1_close == L_1.p2)].append(con)
                connected_segments[L_n][int(L_n_close == L_n.p2)].append(con)

    return connected_segments
