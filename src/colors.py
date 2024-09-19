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
        super().__init__((40, 40, 40), 'neutral', None)

class White(Color):
    def __init__(self):
        super().__init__((255, 255, 255), 'neutral', None)

class Red(Color):
    def __init__(self):
        super().__init__((255, 54, 54), 'warm', None)

class Orange(Color):
    def __init__(self):
        super().__init__((255, 165, 0), 'warm', None)

class Yellow(Color):
    def __init__(self):
        super().__init__((255, 255, 0), 'warm', None)

class Pink(Color):
    def __init__(self):
        super().__init__((255, 90, 190), 'warm', None)

class Brown(Color):
    def __init__(self):
        super().__init__((154, 84, 54), 'warm', None)

class Blue(Color):
    def __init__(self):
        super().__init__((31, 255, 121), 'cool', None)

class Green(Color):
    def __init__(self):
        super().__init__((0, 204, 0), 'cool', None)

class Cyan(Color):
    def __init__(self):
        super().__init__((0, 255, 255), 'cool', None)

class Navy(Color):
    def __init__(self):
        super().__init__((40, 40, 105), 'cool', None)

class Purple(Color):
    def __init__(self):
        super().__init__((123, 104, 245), 'cool', None)

class Black(Color):
    def __init__(self):
        super().__init__((0, 0, 0), 'neutral', None)

COLORS = [Red(), Orange(), Yellow(), Pink(), Brown(), Blue(), Green(), Cyan(), Navy(), Purple()]

HIDDEN=Hidden()
NC=White()

ALL = COLORS + [HIDDEN, NC]