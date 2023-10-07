# Fraction n/d
class Frac:
    def __init__(self, n, d):
        self._n = int(n)
        self._d = int(d)
        self._simplify()

    def _simplify(self):
        import math
        gcd = math.gcd(self._n, self._d)
        self._n //= gcd
        self._d //= gcd

    def n(self):
        return self._n

    def d(self):
        return self._d

    def invert(self):
        return Frac(self._d, self._n)
    
    def __str__(self):
        return "{}/{}".format(self._n, self._d)

    def averageWith(self, other):
        n = self._n * other._d + self._d * other._n
        d = 2 * self._d * other._d
        return Frac(n, d)

def harmonic_mean(a, b):
    f1 = Frac(1, a)
    f2 = Frac(1, b)
    avg = f1.averageWith(f2)
    return avg.invert()
    
a = 0
b = 0
for i in range(2):
    if i == 0:
        a = input()
    else:
        b = input()
result = harmonic_mean(a, b)
print("Harmonic mean of {} and {}: {}".format(a, b, result))