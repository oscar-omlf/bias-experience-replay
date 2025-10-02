
class LinearSchedule:
    def __init__(self, start: float, end: float, duration: int):
        self.start = float(start)
        self.end = float(end)
        self.duration = max(1, int(duration))

    def __call__(self, t: int) -> float:
        frac = min(1.0, max(0.0, t / self.duration))
        return self.start + frac * (self.end - self.start)
