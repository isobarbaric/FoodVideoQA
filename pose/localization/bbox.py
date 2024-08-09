import numpy as np
from dataclasses import dataclass
from strenum import StrEnum


@dataclass
class BoundingBox:
    bounding_box: np.ndarray
    label: str
    score: float

    @property
    def xmin(self) -> float:
        return self.bounding_box[0]

    @property
    def ymin(self) -> float:
        return self.bounding_box[1]

    @property
    def xmax(self) -> float:
        return self.bounding_box[2]

    @property
    def ymax(self) -> float:
        return self.bounding_box[3]

    @property
    def center(self) -> tuple[int, int]:
        xmin, ymin, xmax, ymax = self.bounding_box
        x_mid = (xmin + xmax) / 2
        y_mid = (ymin + ymax) / 2
        return x_mid, y_mid

    @property
    def area(self) -> float:
        return (self.xmax - self.xmin) * (self.ymax - self.ymin)


class Labels(StrEnum):
    MOUTH = "mouth"
    FOOD = "food"
