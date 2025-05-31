# --- START OF FILE points_array.py ---
class PointsArray:
    def __init__(self):
        self.Points = {}
        self._init_points()

    def _init_points(self):
        # Copied directly from your main.py
        self.Points = {
            0: type('Point', (), {'Start': 0, 'End': 0})(),
            1: type('Point', (), {'Start': 1, 'End': 22})(), 2: type('Point', (), {'Start': 1, 'End': 24})(),
            3: type('Point', (), {'Start': 1, 'End': 26})(), 4: type('Point', (), {'Start': 1, 'End': 28})(),
            5: type('Point', (), {'Start': 1, 'End': 30})(), 6: type('Point', (), {'Start': 1, 'End': 32})(),
            7: type('Point', (), {'Start': 1, 'End': 34})(), 8: type('Point', (), {'Start': 1, 'End': 36})(),
            9: type('Point', (), {'Start': 1, 'End': 38})(), 10: type('Point', (), {'Start': 1, 'End': 40})(),
            11: type('Point', (), {'Start': 1, 'End': 42})(), 12: type('Point', (), {'Start': 1, 'End': 44})(),
            13: type('Point', (), {'Start': 1, 'End': 44})(), 14: type('Point', (), {'Start': 1, 'End': 44})(),
            15: type('Point', (), {'Start': 1, 'End': 44})(), 16: type('Point', (), {'Start': 11, 'End': 44})(),
            17: type('Point', (), {'Start': 15, 'End': 44})(), 18: type('Point', (), {'Start': 17, 'End': 44})(),
            19: type('Point', (), {'Start': 20, 'End': 44})(), 20: type('Point', (), {'Start': 21, 'End': 44})(),
            21: type('Point', (), {'Start': 23, 'End': 44})(), 22: type('Point', (), {'Start': 24, 'End': 44})(),
            23: type('Point', (), {'Start': 24, 'End': 44})(), 24: type('Point', (), {'Start': 25, 'End': 44})(),
            25: type('Point', (), {'Start': 25, 'End': 44})(), 26: type('Point', (), {'Start': 25, 'End': 44})(),
            27: type('Point', (), {'Start': 25, 'End': 44})(), 28: type('Point', (), {'Start': 25, 'End': 44})(),
            29: type('Point', (), {'Start': 25, 'End': 44})(), 30: type('Point', (), {'Start': 24, 'End': 44})(),
            31: type('Point', (), {'Start': 24, 'End': 44})(), 32: type('Point', (), {'Start': 23, 'End': 44})(),
            33: type('Point', (), {'Start': 21, 'End': 44})(), 34: type('Point', (), {'Start': 20, 'End': 44})(),
            35: type('Point', (), {'Start': 17, 'End': 44})(), 36: type('Point', (), {'Start': 15, 'End': 44})(),
            37: type('Point', (), {'Start': 11, 'End': 44})(), 38: type('Point', (), {'Start': 1, 'End': 44})(),
            39: type('Point', (), {'Start': 1, 'End': 44})(), 40: type('Point', (), {'Start': 1, 'End': 44})(),
            41: type('Point', (), {'Start': 1, 'End': 44})(), 42: type('Point', (), {'Start': 1, 'End': 42})(),
            43: type('Point', (), {'Start': 1, 'End': 40})(), 44: type('Point', (), {'Start': 1, 'End': 38})(),
            45: type('Point', (), {'Start': 1, 'End': 36})(), 46: type('Point', (), {'Start': 1, 'End': 34})(),
            47: type('Point', (), {'Start': 1, 'End': 32})(), 48: type('Point', (), {'Start': 1, 'End': 30})(),
            49: type('Point', (), {'Start': 1, 'End': 28})(), 50: type('Point', (), {'Start': 1, 'End': 26})(),
            51: type('Point', (), {'Start': 1, 'End': 24})(), 52: type('Point', (), {'Start': 1, 'End': 22})()
        }

    def is_valid(self, col, row): # col: 0-51, row: 0-43
        if col in self.Points:
            point_range = self.Points[col]
            # row is 0-indexed, Point Start/End are 1-indexed
            return point_range.Start <= row + 1 <= point_range.End
        return False
# --- END OF FILE points_array.py ---