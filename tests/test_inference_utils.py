import numpy as np
import pytest
from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Polygon

from Inference.inference_utils import (
    largest_contour,
    largest_polygon,
    load_boundary_points,
    sample_partial_indices,
    set_random_seed,
)
from utils import check_room_shape, visualize_partial_input


def test_check_room_shape_returns_the_adjusted_points():
    points = np.array([[10.0, 10.0], [12.0, 10.0], [12.0, 12.0], [10.0, 12.0]])

    adjusted = check_room_shape(points)

    assert adjusted is points
    assert adjusted.shape == (4, 2)
    assert not np.array_equal(adjusted, np.array([[10.0, 10.0], [12.0, 10.0], [12.0, 12.0], [10.0, 12.0]]))


@pytest.mark.parametrize("ratio, expected_count", [(0.0, 0), (0.25, 2), (1.0, 7)])
def test_partial_sampling_includes_the_full_room_population(ratio, expected_count):
    indices = sample_partial_indices(7, ratio, np.random.RandomState(3))

    assert len(indices) == expected_count
    assert np.all((1 <= indices) & (indices <= 7))


@pytest.mark.parametrize("ratio", [-0.01, 1.01])
def test_partial_sampling_rejects_invalid_ratios(ratio):
    with pytest.raises(ValueError, match="between 0 and 1"):
        sample_partial_indices(7, ratio, np.random.RandomState(3))


def test_largest_polygon_ignores_empty_and_non_polygonal_components():
    geometry = GeometryCollection(
        [LineString([(0, 0), (1, 1)]), Polygon(), Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])]
    )

    polygon = largest_polygon(geometry)

    assert polygon is not None
    assert polygon.area == 16
    assert largest_polygon(GeometryCollection()) is None

    multipolygon = MultiPolygon(
        [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(0, 0), (3, 0), (3, 3), (0, 3)]),
        ]
    )
    assert largest_polygon(multipolygon).area == 9


def test_largest_contour_ignores_empty_and_degenerate_contours():
    degenerate = np.array([[[0, 0]], [[1, 1]]], dtype=np.int32)
    valid = np.array([[[0, 0]], [[4, 0]], [[4, 4]], [[0, 4]]], dtype=np.int32)

    assert largest_contour([]) is None
    assert largest_contour([degenerate]) is None
    np.testing.assert_array_equal(largest_contour([degenerate, valid]), valid)


def test_missing_boundary_has_an_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"extract.*0\.7z|step \(a\)"):
        load_boundary_points(tmp_path / "missing.png")


def test_optional_seed_controls_numpy_and_tensorflow():
    class RandomAPI:
        value = None

        @classmethod
        def set_seed(cls, value):
            cls.value = value

    class TensorFlowStub:
        random = RandomAPI

    set_random_seed(17, TensorFlowStub)
    first = np.random.randint(0, 1000, size=5)
    set_random_seed(17, TensorFlowStub)
    second = np.random.randint(0, 1000, size=5)

    np.testing.assert_array_equal(first, second)
    assert RandomAPI.value == 17


def test_partial_input_visualization_does_not_need_an_external_canvas(tmp_path):
    output = tmp_path / "partial.png"
    adjacency = np.zeros((10, 8), dtype=np.uint8)
    adjacency[1, 1] = 1

    canvas = visualize_partial_input(
        np.array([1, 8]), np.array([2]), adjacency, np.array([3]), np.array([4]), output
    )

    assert output.exists()
    assert canvas.shape == (80, 130)
    assert canvas[0:10, 0:10].max() == 0
    assert canvas[70:80, 0:10].max() == 0
    assert canvas[10:20, 10:20].max() == 0
