import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import compiler
import ctypes
import error
from loma import Array, In, Out

def declaration_float() -> float:
    x : float = 5
    return x

def declaration_int() -> int:
    x : int = 4
    return x

def test_declaration():
    lib = compiler.compile(declaration_float)
    assert abs(lib.declaration_float() - 5) < 1e-6
    lib = compiler.compile(declaration_int)
    assert lib.declaration_int() == 4

def binaryops() -> float:
    x : float = 5.0
    y : float = 6.0
    a : float = x + y
    b : float = a - x
    c : float = b * y
    d : float = c / a
    return d

def test_binary_ops():
    lib = compiler.compile(binaryops)
    # a = x + y = 5 + 6 = 11
    # b = a - x = 11 - 5 = 6
    # c = b * y = 6 * 6 = 36
    # d = c / a = 36 / 11
    assert abs(lib.binaryops() - 36.0 / 11.0) < 1e-6

def args(x : In[float], y : In[int]) -> int:
    z : int = x
    return z + y

def test_args():
    lib = compiler.compile(args)
    assert lib.args(4.5, 3) == 7

def mutation() -> float:
    a : float = 5.0
    a = 6.0
    return a

def test_mutation():
    lib = compiler.compile(mutation)
    assert abs(lib.mutation() - 6) < 1e-6

def array_read(x : In[Array[float]]) -> float:
    return x[0]

def test_array_read():
    lib = compiler.compile(array_read)
    py_arr = [1.0, 2.0]
    arr = (ctypes.c_float * len(py_arr))(*py_arr)
    assert lib.array_read(arr) == 1.0

def array_write(x : Out[Array[float]]):
    x[0] = 2.0

def test_array_write():
    lib = compiler.compile(array_write)
    py_arr = [0.0, 0.0]
    arr = (ctypes.c_float * len(py_arr))(*py_arr)
    lib.array_write(arr)
    assert arr[0] == 2.0

def compare(x : In[int], y : In[int],
            out : Out[Array[int]]):
    out[0] = x < y
    out[1] = x <= y
    out[2] = x > y
    out[3] = x >= y
    out[4] = x == y
    out[5] = x < y and x > y
    out[6] = x < y or x > y

def test_compare():
    lib = compiler.compile(compare)
    py_arr = [0] * 7
    arr = (ctypes.c_int * len(py_arr))(*py_arr)
    # 5 < 6 : True
    # 5 <= 6 : True
    # 5 > 6 : False
    # 5 >= 6 : False
    # 5 == 6 : False
    lib.compare(5, 6, arr)
    assert arr[0] != 0
    assert arr[1] != 0
    assert arr[2] == 0
    assert arr[3] == 0
    assert arr[4] == 0
    assert arr[5] == 0
    assert arr[6] != 0
    # 5 < 5 : False
    # 5 <= 5 : True
    # 5 > 5 : False
    # 5 >= 5 : True
    # 5 == 5 : True
    lib.compare(5, 5, arr)
    assert arr[0] == 0
    assert arr[1] != 0
    assert arr[2] == 0
    assert arr[3] != 0
    assert arr[4] != 0
    assert arr[5] == 0
    assert arr[6] == 0

def if_else(x : In[float]) -> float:
    z : float = 0.0
    if x > 0:
        z = 4.0
    else:
        z = -4.0
    return z

def test_if_else():
    lib = compiler.compile(if_else)
    assert lib.if_else(0.5) == 4.0
    assert lib.if_else(-0.5) == -4.0

def duplicate_declare() -> float:
    x : float = 5
    x : float = 6
    return 0

def test_duplicate_declare():
    try:
        lib = compiler.compile(duplicate_declare)
    except error.DuplicateVariable as e:
        assert e.var == 'x'
        assert e.first_lineno == 2
        assert e.duplicate_lineno == 3

def undeclared_var() -> float:
    a : float = 5.0
    b = 6.0
    return a

def test_undeclared_var():
    try:
        lib = compiler.compile(undeclared_var)
    except error.UndeclaredVariable as e:
        assert e.var == 'b'
        assert e.lineno == 3

if __name__ == '__main__':
    test_declaration()
    test_binary_ops()
    test_args()
    test_mutation()
    test_array_read()
    test_array_write()
    test_compare()
    test_if_else()

    # test compile errors
    test_duplicate_declare()
    test_undeclared_var()
