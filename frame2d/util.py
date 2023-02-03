from . import frame2d


def boxcolumn(b, h, t, i, j, pin = 0, add_mass = 0):

    A = b * h - (b - 2 * t) * (h - 2 * t)
    I = (b * h**3 - (b - 2 * t) * (h - 2 * t)**3) / 12
    E = 205000000000
    G = 79000000000
    m = 7850
    k = A / (2 * (h - 2 * t) * t)
    memi = frame2d.Beam2d(E, G, m, A, I, i, j, k = k, pin = pin, add_mass = add_mass)
    
    return memi


def hbeam(b, h, t1, t2, i, j, pin = 0, add_mass = 0):

    A = b * h - (b - t1) * (h - 2 * t2)
    I = (b * h**3 - (b - t1) * (h - 2 * t2)**3) / 12
    E = 205000000000
    G = 79000000000
    m = 7850
    k = A / (t1 * (h - 2 * t2))
    memi = frame2d.Beam2d(E, G, m, A, I, i, j, k = k, pin = pin, add_mass = add_mass)
    
    return memi