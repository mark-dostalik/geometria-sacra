"""
Script for generating a random maze of given size using Prim's algorithm, or recursive backtracking.

All generated intermediate maze states are stored on disk as 4K images.
"""
import argparse
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Final

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class Maze(ABC):

    IN: Final[int] = 1
    REMOVED_WALL: Final[int] = 2

    def __init__(self, height: int, output_folder_path: Path):
        """Initialize with odd maze height and output folder path."""
        if height % 2 == 0:
            raise NotImplementedError('Height must be an odd number.')

        self.height = height
        self.width = self._get_optimal_width()
        self.seed = self._locate_seed()
        self.maze = np.zeros((self.height, self.width), dtype=np.uint8)
        self.image = np.zeros_like(self.maze, dtype=bool)
        self.num_nodes = ((self.width - 1) // 2 * (self.height - 1) // 2)
        self.output_folder_path = output_folder_path

    def _get_optimal_width(self) -> int:
        """Compute odd maze width so that the ratio between width and height is the closest to 16:9 aspect ratio."""
        floor, ceil = int(np.floor((16 / 9) * self.height)), int(np.ceil((16 / 9) * self.height))
        if floor == ceil:
            return floor - 1  # subtract one to get an odd number
        if ceil % 2 == 0:
            return floor
        return ceil

    def _locate_seed(self) -> tuple[int, int]:
        """Locate initial seed coordinates as close to the center of the maze as possible."""
        seed_x, seed_y = (self.width - 1) // 2, (self.height - 1) // 2
        seed_x = seed_x if seed_x % 2 != 0 else seed_x + 1
        seed_y = seed_y if seed_y % 2 != 0 else seed_y + 1
        return seed_x, seed_y

    def _is_valid(self, x: int, y: int) -> bool:
        return (0 < x < (self.width - 1)) and (0 < y < (self.height - 1))

    def _get_adjacent_cells(self, x: int, y: int) -> list[tuple[int, int]]:
        return [(_x, _y) for _x, _y in ((x + 2, y), (x - 2, y), (x, y + 2), (x, y - 2)) if self._is_valid(_x, _y)]

    def _get_unmarked_neighbors(self, x: int, y: int) -> list[tuple[int, int]]:
        return [(_x, _y) for _x, _y in self._get_adjacent_cells(x, y) if self.maze[_y, _x] != self.IN]

    def _get_marked_neighbors(self, x: int, y: int) -> list[tuple[int, int]]:
        return [(_x, _y) for _x, _y in self._get_adjacent_cells(x, y) if self.maze[_y, _x] == self.IN]

    def _remove_wall(self, x: int, y: int, x_prev: int, y_prev: int):
        self.maze[(y + y_prev) // 2, (x + x_prev) // 2] = self.REMOVED_WALL
        self.image[(y + y_prev) // 2, (x + x_prev) // 2] = True

    def _save_state(self, i: int) -> None:
        """Save the current state as an image of 4K resolution."""
        fig = plt.figure(figsize=(16 / 9, 1))
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(self.image, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "#00CD00"]))
        file_stem = (len(str(self.num_nodes)) - len(str(i))) * '0' + str(i)
        output_path = f'{self.output_folder_path}/{file_stem}.png'
        plt.savefig(output_path, dpi=2160)
        plt.close(fig)

    @abstractmethod
    def run(self):
        """Produce 4K images of maze generation intermediate states."""
        pass


class PrimMaze(Maze):
    """Generates a random maze using Prim's algorithm and saves all the intermediate maze states as 4K images."""

    FRONTIER: Final[int] = 3

    def __init__(self, height, output_folder_path):
        super().__init__(height, output_folder_path)
        self.frontier: list[tuple[int, int]] = []

    def _add_to_frontier(self, x: int, y: int) -> None:
        if self._is_valid(x, y) and self.maze[y, x] == 0:
            self.maze[y, x] = self.FRONTIER
            self.frontier.append((x, y))

    def _mark_in(self, x: int, y: int) -> None:
        self.maze[y, x] = self.IN
        self.image[y, x] = True
        for _x, _y in self._get_adjacent_cells(x, y):
            self._add_to_frontier(_x, _y)

    def run(self):
        self._save_state(0)
        self._mark_in(*self.seed)

        i = 1
        with tqdm(total=self.num_nodes) as pbar:
            while self.frontier:
                self._save_state(i)
                pbar.update()
                x, y = self.frontier.pop(np.random.randint(0, len(self.frontier)))
                neighbors = self._get_marked_neighbors(x, y)
                x_neighbor, y_neighbor = neighbors.pop(np.random.randint(0, len(neighbors)))
                self._remove_wall(x, y, x_neighbor, y_neighbor)
                self._mark_in(x, y)
                i += 1

            pbar.update()
            self._save_state(i)


class RecursiveBacktrackingMaze(Maze):
    """Generates a random maze using recursive backtracking and saves all the intermediate maze states as 4K images."""

    def __init__(self, height, output_folder_path):
        super().__init__(height, output_folder_path)
        self.stack: list[tuple[int, int]] = []

    def _mark_in(self, x: int, y: int) -> None:
        self.maze[y, x] = self.IN
        self.image[y, x] = True
        self.stack.append((x, y))

    def run(self):
        self._save_state(0)
        self._mark_in(*self.seed)

        i = 1
        with tqdm(total=self.num_nodes) as pbar:
            while self.stack:
                self._save_state(i)
                x_prev, y_prev = self.stack[-1]
                unmarked_neighbors = self._get_unmarked_neighbors(x_prev, y_prev)
                if not unmarked_neighbors:
                    self.stack.pop()
                    continue
                pbar.update()
                x, y = unmarked_neighbors.pop(np.random.randint(0, len(unmarked_neighbors)))
                self._remove_wall(x, y, x_prev, y_prev)
                self._mark_in(x, y)
                i += 1

            pbar.update()
            self._save_state(i)


def main():
    parser = get_parser()
    options = parser.parse_args()
    output_folder_path = Path(options.output_path)
    output_folder_path.mkdir(exist_ok=True, parents=True)

    match options.method:
        case 'prim':
            maze = PrimMaze(int(options.height), output_folder_path)
        case 'recursive-backtracking':
            maze = RecursiveBacktrackingMaze(int(options.height), output_folder_path)
        case _:
            raise NotImplementedError

    maze.run()


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        'height',
        type=str,
        help="Number of path rows in the generated maze. Must be an odd number.",
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Path to a folder to store generated maze state images to.",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["prim", "recursive-backtracking"],
        default="prim",
        help="Algorithm to be used for random maze generation."
    )
    return parser


if __name__ == '__main__':
    main()
