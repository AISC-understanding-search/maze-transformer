import numpy as np
from numpy.testing import assert_array_equal

from maze_transformer.generation.latticemaze import LatticeMaze
from maze_transformer.generation.utils import bool_array_from_string


def test_as_img():
    connection_list = bool_array_from_string(
        """
        F T T
        T F T
        F F F

        T T F
        T F F
        T F F
        """,
        shape=[2, 3, 3],
    )

    img = LatticeMaze(connection_list=connection_list).as_img()

    expected = bool_array_from_string(
        """
        x x x x x x x
        x _ _ _ _ _ x
        x x x _ x _ x
        x _ _ _ x _ x
        x _ x x x _ x
        x _ _ _ x _ x
        x x x x x x x
        """,
        shape=[7, 7],
        true_symbol="_",
    )

    assert_array_equal(expected, img)

def test_as_adjlist():
    connection_list = bool_array_from_string(
        """
        F T
        F F

        T F
        T F
        """,
        shape=[2, 2, 2],
    )

    maze = LatticeMaze(connection_list=connection_list)


    adjlist = maze.as_adjlist(shuffle_d0=False, shuffle_d1=False)

    expected = [
        [[0,1], [1,1]],
        [[0,0], [0,1]],
        [[1,0], [1,1]]
    ]

    assert _to_nested_set(expected) == _to_nested_set(adjlist)

# We don't care about order of coordinate pairs within
# the adjlist or coordinates within each coordinate pair.
def _to_nested_set(nested_list):
    return {
        frozenset([tuple(start_coord), tuple(end_coord)]) for start_coord, end_coord in nested_list
    }