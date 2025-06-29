from dataclasses import dataclass


@dataclass
class Scaling:
    factor: float
    offset: float

    def get_unscale_scaling(self):
        return Scaling(1 / self.factor, -self.offset / self.factor)

    def scale(self, value):
        return (value - self.offset) / self.factor

    def unscale(self, value):
        return value * self.factor + self.offset

    def scale_derivate(self, value):
        return value / self.factor

    def unscale_derivate(self, value):
        return value * self.factor

    @staticmethod
    def from_range(min: float, max: float):
        factor = (max - min) / 2
        offset = (max + min) / 2

        return Scaling(factor, offset)
