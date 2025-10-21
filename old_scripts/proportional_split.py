def proportional_split(x: float, y: float, z: float) -> tuple[float, float]:
    a: float = 1.0 - x
    b: float = y / a

    return (round(a, 10), round(b, 10))
