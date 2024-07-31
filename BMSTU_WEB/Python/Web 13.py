class MyClass(object):
    def fib(n):
        return MyClass.fib(n - 1) + MyClass.fib(n - 2) if n > 1 else n
print(MyClass.fib(6))

import sys
thismodule = sys.modules[__name__]
currency = {'rur': 1, 'eur': 70, 'usd': 60 }
# currency = {'gbp: 100, 'eur': 30, 'usd': 20 }
# currency = {'yuan': 1, 'eur': 40, 'usd': 30,'fr': 30}
# currency = {'cny': 20, 'aud': 40, 'usd': 80, 'eur': 100}
def get_function(c1, c2):
    def wrapped(val):
        return val * c1 / c2
    return wrapped
for k1, v1 in currency.items():
    for k2, v2 in currency.items():
        setattr(thismodule, k1 + "_to_" + k2, get_function(v1, v2))

print(usd_to_eur(7))