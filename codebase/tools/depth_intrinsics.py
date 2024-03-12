class DepthIntrinsics:
    def __init__(self):
        self._data = {
            'width': None,
            'height': None,
            'fx': None,
            'fy': None,
            'ppx': None,
            'ppy': None,
            'scale': None
        }

    @property
    def width(self):
        return self._data['width']

    @width.setter
    def width(self, value):
        self._data['width'] = value

    @property
    def height(self):
        return self._data['height']

    @height.setter
    def height(self, value):
        self._data['height'] = value

    @property
    def fx(self):
        return self._data['fx']

    @fx.setter
    def fx(self, value):
        self._data['fx'] = value

    @property
    def fy(self):
        return self._data['fy']

    @fy.setter
    def fy(self, value):
        self._data['fy'] = value

    @property
    def ppx(self):
        return self._data['ppx']

    @ppx.setter
    def ppx(self, value):
        self._data['ppx'] = value

    @property
    def ppy(self):
        return self._data['ppy']

    @ppy.setter
    def ppy(self, value):
        self._data['ppy'] = value

    @property
    def scale(self):
        return self._data['scale']

    @scale.setter
    def scale(self, value):
        self._data['scale'] = value
