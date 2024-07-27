from abc import ABC

class Color(ABC):
    def __init__(self, rgb, tone, encoding):
        self.rgb = rgb
        self.tone = tone
        self.encoding = encoding

    @property
    def name(self):
        return self.__class__.__name__

class Hidden(Color):
    def __init__(self):
        super().__init__((40, 40, 40), "neutral", None)

class White(Color):
    def __init__(self):
        super().__init__((255, 255, 255), "neutral", None)

class Red(Color):
    def __init__(self):
        super().__init__((255, 0, 0), "warm", None)

class Orange(Color):
    def __init__(self):
        super().__init__((255, 165, 0), "warm", None)

class Yellow(Color):
    def __init__(self):
        super().__init__((255, 255, 0), "warm", None)

class Green(Color):
    def __init__(self):
        super().__init__((0, 255, "cool", 0), None)

class Blue(Color):
    def __init__(self):
        super().__init__((0, 0, 255), "cool", None)

class Pink(Color):
    def __init__(self):
        super().__init__((255, 192, 203), "warm", None)

class Violet(Color):
    def __init__(self):
        super().__init__((138, 43, 226), "cool", None)

class Cyan(Color):
    def __init__(self):
        super().__init__((0, 255, 255), "cool", None)

class Coral(Color):
    def __init__(self):
        super().__init__((255, 127, 80), "warm", None)

class Teal(Color):
    def __init__(self):
        super().__init__((0, 128, 128), "cool", None)

class Black(Color):
    def __init__(self):
        super().__init__((0, 0, 0), "neutral", None)

COLORS = [Red(), Orange(), Yellow(), Green(), Blue(), Pink(), Violet(), Cyan(), Coral(), Teal()]
HIDDEN=Hidden()
NC=White()