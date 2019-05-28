import math

import pytest

from lsm import LineSegment, merge_lines, merge_two_lines, angle_difference


def test_linesegment_length():
    l = LineSegment(0, 0, 3, 4)
    assert l.length == 5


def test_linesegment_angle():
    l = LineSegment(0, 0, 5, 0)
    assert l.angle == 0

    l = LineSegment(0, 0, 0, 5)
    assert l.angle == math.pi/2

    l = LineSegment(4, 4, 4, 9)
    assert l.angle == math.pi/2


def test_merge_trivial():
    L = [LineSegment(0, 0, 10, 10), LineSegment(0, 0, 10, 10)]
    assert merge_lines(L, 1, 1) == [LineSegment(0, 0, 10, 10)]


def test_merge_simple():
    L = [LineSegment(0, 0, 10, 10), LineSegment(0, 0, 10, 11)]
    assert merge_lines(L, 1, 1) == [LineSegment(0, 0, 10, 11)]


def test_merge_two_trivial():
    L_1 = LineSegment(0, 0, 10, 10)
    L_2 = LineSegment(0, 0, 10, 10)
    assert merge_two_lines(L_1, L_2, 1, 1) == LineSegment(0, 0, 10, 10)


def test_merge_two_simple():
    L_1 = LineSegment(0, 0, 10, 10)
    L_2 = LineSegment(0, 0, 10, 11)
    assert merge_two_lines(L_1, L_2, 1, 1) == LineSegment(0, 0, 10, 11)


def test_merge_two_separate():
    L_1 = LineSegment(0, 0, 2, 0)
    L_2 = LineSegment(5, 0, 7, 0)
    assert merge_two_lines(L_1, L_2, 1, 1) is None


@pytest.mark.parametrize("a, b, expected", [
    (0, 0, 0),
    (0, -math.pi, 0),
    (-math.pi/2, math.pi/2, 0),
    (-math.pi*(3/4), math.pi*(3/4), math.pi/2),
])
def test_angle_difference(a, b, expected):
    assert angle_difference(a, b) == expected
