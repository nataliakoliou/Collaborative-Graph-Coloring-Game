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
        super().__init__((255, 0, 0), 'warm', 'Red')

class Orange(Color):
    def __init__(self):
        super().__init__((255, 165, 0), 'warm', 'Orange')

class Yellow(Color):
    def __init__(self):
        super().__init__((255, 255, 0), 'warm', 'Yellow')

class Rose(Color):
    def __init__(self):
        super().__init__((209, 95, 128), 'warm', 'Rose')

class Coral(Color):
    def __init__(self):
        super().__init__((255, 127, 80), 'warm', 'Coral')

class Pink(Color):
    def __init__(self):
        super().__init__((255, 171, 205), 'warm', 'Pink')

class Peach(Color):
    def __init__(self):
        super().__init__((255, 180, 92), 'warm', 'Peach')

class Gold(Color):
    def __init__(self):
        super().__init__((255, 255, 80), 'warm', 'Gold')

class Crimson(Color):
    def __init__(self):
        super().__init__((220, 20, 60), 'warm', 'Crimson')

class Scarlet(Color):
    def __init__(self):
        super().__init__((255, 54, 54), 'warm', 'Scarlet')

class Fuchsia(Color):
    def __init__(self):
        super().__init__((255, 90, 190), 'warm', 'Fuchsia')

class Brown(Color):
    def __init__(self):
        super().__init__((154, 84, 54), 'warm', 'Brown')

class Redwood(Color):
    def __init__(self):
        super().__init__((183, 58, 58), 'warm', 'Redwood')

class Lime(Color):
    def __init__(self):
        super().__init__((0, 204, 0), 'cool', 'Lime')

class Blue(Color):
    def __init__(self):
        super().__init__((0, 0, 255), 'cool', 'Blue')

class Green(Color):
    def __init__(self):
        super().__init__((30, 120, 30), 'cool', 'Green')

class Cyan(Color):
    def __init__(self):
        super().__init__((0, 255, 255), 'cool', 'Cyan')

class Teal(Color):
    def __init__(self):
        super().__init__((0, 128, 128), 'cool', 'Teal')

class Azure(Color):
    def __init__(self):
        super().__init__((30, 144, 255), 'cool', 'Azure')

class Mint(Color):
    def __init__(self):
        super().__init__((31, 255, 121), 'cool', 'Mint')

class Cobalt(Color):
    def __init__(self):
        super().__init__((10, 88, 235), 'cool', 'Cobalt')

class Purple(Color):
    def __init__(self):
        super().__init__((85, 0, 255), 'cool', 'Purple')

class Navy(Color):
    def __init__(self):
        super().__init__((10, 58, 100), 'cool', 'Navy')

class Moss(Color):
    def __init__(self):
        super().__init__((57, 131, 74), 'cool', 'Moss')

class Lavender(Color):
    def __init__(self):
        super().__init__((123, 104, 245), 'cool', 'Lavender')

class Sky(Color):
    def __init__(self):
        super().__init__((120, 200, 255), 'cool', 'Sky')

class Black(Color):
    def __init__(self):
        super().__init__((0, 0, 0), 'neutral', None)

COLORS = [Red(), Orange(), Yellow(), Rose(), Coral(), Pink(), Peach(), Gold(), Crimson(),
          Scarlet(), Fuchsia(), Brown(), Redwood(), Lime(), Blue(), Green(), Cyan(),
          Teal(), Azure(), Mint(), Cobalt(), Purple(), Navy(), Moss(), Lavender(), Sky()]

HIDDEN=Hidden()
NC=White()

ALL = COLORS + [HIDDEN, NC]