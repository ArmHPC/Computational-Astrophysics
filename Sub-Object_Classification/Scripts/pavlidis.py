"""
    pavlidis.py - Python implementation of Pavlidis algorithm
"""

import numpy as np


def get_pixel(array: np.ndarray, row: int, column: int):
    if row < 0 or column < 0:
        return False
    if row >= array.shape[0] or column >= array.shape[1]:
        return False
    return array[row, column]


def pavlidis_impl(array: np.ndarray, seed_row: int, seed_column: int, initial_dir: int):
    n_turns: int = 0
    current: np.ndarray = np.array([seed_row, seed_column])
    coords: list = [current.copy()]
    direction: int = initial_dir

    while True:
        if direction == 0:  # -x
            if get_pixel(array, current[0] + 1, current[1] - 1):
                direction = 3
                current[0] += 1
                current[1] -= 1
            elif get_pixel(array, current[0], current[1] - 1):
                current[1] -= 1
            elif get_pixel(array, current[0] - 1, current[1] - 1):
                current[0] -= 1
                current[1] -= 1
            else:
                direction = 1
                n_turns += 1
                if n_turns == 4:
                    break
                continue
        elif direction == 1:  # -y
            if get_pixel(array, current[0] - 1, current[1] - 1):
                direction = 0
                current[0] -= 1
                current[1] -= 1
            elif get_pixel(array, current[0] - 1, current[1]):
                current[0] -= 1
            elif get_pixel(array, current[0] - 1, current[1] + 1):
                current[0] -= 1
                current[1] += 1
            else:
                direction = 2
                n_turns += 1
                if n_turns == 4:
                    break
                continue
        elif direction == 2:  # +x
            if get_pixel(array, current[0] - 1, current[1] + 1):
                direction = 1
                current[0] -= 1
                current[1] += 1
            elif get_pixel(array, current[0], current[1] + 1):
                current[1] += 1
            elif get_pixel(array, current[0] + 1, current[1] + 1):
                current[0] += 1
                current[1] += 1
            else:
                direction = 3
                n_turns += 1
                if n_turns == 4:
                    break
                continue
        else:  # +y
            if get_pixel(array, current[0] + 1, current[1] + 1):
                direction = 2
                current[0] += 1
                current[1] += 1
            elif get_pixel(array, current[0] + 1, current[1]):
                current[0] += 1
            elif get_pixel(array, current[0] + 1, current[1] - 1):
                current[0] += 1
                current[1] -= 1
            else:
                direction = 0
                n_turns += 1
                if n_turns == 4:
                    break
                continue
        n_turns = 0
        if current[0] == seed_row and current[1] == seed_column:
            break
        if len(coords) >= 9000:
            break
        coords.append(current.copy())
    return np.array(coords, dtype=np.uint64)


def pavlidis(array: np.ndarray, seed_row: int, seed_column: int):
    """Find the boundary around an object by walking in a clockwise direction

    Only works on 4-connected segments.
    Start walking from the seed pixel.

    :param array: a boolean 2d array of foreground pixels
    :param seed_row: the row (1st) coordinate of the seed pixel
    :param seed_column: the column (2nd) coordinate of the seed pixel
    """

    if not get_pixel(array, seed_row, seed_column):
        raise AssertionError("Seed pixel is not within an object")
    #
    # Check for interior pixel
    #
    if get_pixel(array, seed_row - 1, seed_column) and \
            get_pixel(array, seed_row, seed_column + 1) and \
            get_pixel(array, seed_row + 1, seed_column) and \
            get_pixel(array, seed_row, seed_column - 1):
        raise AssertionError("Seed pixel is in interior of object")

    #
    # Set the initial direction so that p1 is not true.
    #
    direction = 0
    if get_pixel(array, seed_row + 1, seed_column - 1):
        direction = 1
        if get_pixel(array, seed_row - 1, seed_column - 1):
            direction = 2
            if get_pixel(array, seed_row - 1, seed_column + 1):
                direction = 3

    return pavlidis_impl(array, seed_row, seed_column, direction)
