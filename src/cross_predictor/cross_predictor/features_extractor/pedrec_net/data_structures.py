from ..utils.dataclasses import dataclass

@dataclass
class ImageSize(object):
    width: int
    height: int

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, ImageSize):
            return self.width == other.width and \
                   self.height == other.height
        return False
