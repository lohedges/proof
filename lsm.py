import math

import numpy as np
from scipy.spatial.distance import pdist


class LineSegment:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

        self._length = None
        self._angle = None

    @property
    def length(self):
        if self._length is None:
            self._length = math.sqrt(pow(self.x2 - self.x1, 2) + pow(self.y2 - self.y1, 2))
        return self._length

    @property
    def angle(self):
        if self._angle is None:
            self._angle = math.atan2(self.y2 - self.y1, self.x2 - self.x1)
        return self._angle

    def __repr__(self):
        return f"LineSegment({self.x1}, {self.y1}, {self.x2}, {self.y2})"

    def __eq__(self, other):
        return (self.x1, self.y1, self.x2, self.y2) == (other.x1, other.y1, other.x2, other.y2)

    def __hash__(self):
        return hash((self.x1, self.y1, self.x2, self.y2))


def angle_difference(a, b):
    r = a - b
    r = (r + math.pi/2) % math.pi - math.pi/2
    return abs(r)


def merge_lines(L, tau_theta, xi_s):
    while True:
        n = len(L)
        L = sorted(L, key=lambda x: x.length, reverse=True)

        for i, L_1 in enumerate(L):
            l1 = L_1.length
            tau_s = xi_s * l1
            P = [l for l in L if l is not L_1]
            P = filter_by_angle(P, L_1, tau_theta)
            P = filter_by_position(P, L_1, tau_s)

            R = []
            for L_2 in P:
                M = merge_two_lines(L_1, L_2, xi_s, tau_theta)

                if M is not None:
                    L[i] = M
                    L_1 = M
                    R.append(L_2)
            # The following is slow and bad. We should find a different way
            # Probably give all segments a unique ID and match by that
            for to_remove in R:
                for j, l in enumerate(L):
                    if l is to_remove:
                        del L[j]
        if len(L) == n:
            # If we didn't remove any segments this time, we're done
            break
    return L


def filter_by_angle(L, L1, tau_theta):
    for L2 in L:
        if angle_difference(L2.angle, L1.angle) < tau_theta:
            yield L2


def filter_by_position(L, L1, tau_s):
    for L2 in L:
        if (
                abs(L1.x1 - L2.x1) < tau_s or
                abs(L1.x1 - L2.x2) < tau_s or
                abs(L1.x2 - L2.x1) < tau_s or
                abs(L1.x2 - L2.x2) < tau_s
        ) and (
                abs(L1.y1 - L2.y1) < tau_s or
                abs(L1.y1 - L2.y2) < tau_s or
                abs(L1.y2 - L2.y1) < tau_s or
                abs(L1.y2 - L2.y2) < tau_s
        ):
            yield L2


def merge_two_lines(L_1, L_2, xi_s, tau_theta):
    l_1 = abs(L_1.length)
    l_2 = abs(L_2.length)
    theta_1 = L_1.angle
    theta_2 = L_2.angle

    if l_1 < l_2:
        L_1, L_2 = L_2, L_1
        l_1, l_2 = l_2, l_1
        theta_1, theta_2 = theta_2, theta_1

    d = closest_end_points(L_1, L_2)

    tau_s = xi_s * l_1

    if d > tau_s:
        M = None
        return M

    tau_theta_star = adaptive_spatial_proximity_threshold(tau_theta, l_1, l_2, d, tau_s)

    theta = angle_difference(theta_2, theta_1)

    if theta < tau_theta_star or theta > (math.pi - tau_theta_star):
        M = extreme_points(L_1, L_2)
        theta_M = M.angle
        if angle_difference(theta_1, theta_M) > 0.5 * tau_theta:
            M = None
    else:
        M = None

    return M


def closest_end_points(L_1, L_2):
    contenders = []
    contenders.append(LineSegment(L_1.x1, L_1.y1, L_2.x1, L_2.y1))
    contenders.append(LineSegment(L_1.x1, L_1.y1, L_2.x2, L_2.y2))
    contenders.append(LineSegment(L_1.x2, L_1.y2, L_2.x1, L_2.y1))
    contenders.append(LineSegment(L_1.x2, L_1.y2, L_2.x2, L_2.y2))

    shortest = min(contenders, key=lambda x: x.length)

    return shortest.length


def adaptive_spatial_proximity_threshold(tau_theta, l_1, l_2, d, tau_s):
    l_2_hat = l_2 / l_1
    d_hat = d / tau_s
    lambd = l_2_hat + d_hat

    tau_theta_star = tau_theta * (1 - (1 / (1 + math.exp(-2 * (lambd - 1.5)))))

    return tau_theta_star


def extreme_points(L_1, L_2):
    X = np.array([
        [L_1.x1, L_1.y1],
        [L_1.x2, L_1.y2],
        [L_2.x1, L_2.y1],
        [L_2.x2, L_2.y2]
    ])
    max_pos = pdist(X).argmax()
    coords = {
        0: (L_1.x1, L_1.y1, L_1.x2, L_1.y2),
        1: (L_1.x1, L_1.y1, L_2.x1, L_2.y1),
        2: (L_1.x1, L_1.y1, L_2.x2, L_2.y2),
        3: (L_1.x2, L_1.y2, L_2.x1, L_2.y1),
        4: (L_1.x2, L_1.y2, L_2.x2, L_2.y2),
        5: (L_2.x1, L_2.y1, L_2.x2, L_2.y2),
    }[max_pos]
    return LineSegment(*coords)
