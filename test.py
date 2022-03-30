import unittest
from libreria import *
'''TEST LIBRERIA JUAN JOSE ALVAREZ'''
class TestStringMethods(unittest.TestCase):
    def test_Estados_Posibles(self):
        m = [[0,1/2],[1/2,0]]
        m2 = [[0,-1j],[1j,0]]
        m3 = [[1,0],[0,1]]
        self.assertEqual(valores_esperados(m,1),([0.5, -0.5]))
        self.assertEqual(valores_esperados(m2,1),([(1+0j), (-1+0j)]))
        self.assertEqual(valores_esperados(m3,1),([1.0, 1.0]))
    def test_probabilidad_transicion(self):
        v1 = [[1,0],[0,0]]
        v2 = [[0,0],[1,0]]
        sx = [[0,1/2],[1/2,0]]
        sy = [[0,-1j],[1j,0]]
        sz = [[1,0],[0,-1]]
        self.assertEqual(probabilidad(v1,v2,sx,1),([[[0.71, 0.0]]]))
        self.assertEqual(probabilidad(v1,v2,sx,2),([[[0.71, 0.0]]]))
        self.assertEqual(probabilidad(v1,v2,sx,3),([[[-0.71, 0.0]]]))
        self.assertEqual(probabilidad(v1,v2,sx,4),([[[0.71, 0.0]]]))
    def test_Producto(self):
        m1 = [[(0,0),(1,0)],[(1,0),(0,0)]]
        m2 = [[((2**(1/2))/2,0),((2**(1/2))/2,0)],[((2**(1/2))/2,0),(-(2**(1/2))/2,0)]]
        self.assertEqual(comprobar_producto(m1,m2),(False))
    def test_probabilidad(self):
        m1 = [[(0,0),(1/(2**(1/2)),0),(1/(2**(1/2)),0),(0,0)],
          [(0,1/(2**(1/2))),(0,0),(0,0),(1/(2**(1/2)),0)],
          [(1/(2**(1/2)),0),(0,0),(0,0),(0,1/(2**(1/2)))],
            [(0,0),(1/(2**(1/2)),0),(0,-1/(2**(1/2))),(0,0)]]
        v = [[1,0],[0,0],[0,0],[0,0]]
        self.assertEqual(probabilidad_3_clic(m1,v),(0.25))
if __name__ == '__main__':
    unittest.main()
