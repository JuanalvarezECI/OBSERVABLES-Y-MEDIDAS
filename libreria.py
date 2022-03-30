from sys import stdin
from fractions import Fraction
import math
import numpy as np

'''LIBRERIA 4.3.1, 4.3.2, 4.4.1, 4.4.2 , 29/3/22 JUAN JOSE ALVAREZ BELTRAN'''
def sumComplejos(a,b):
    return (a[0]+b[0], a[1]+b[1])
def Restacomplejos(a,b):
    return (a[0]-b[0], a[1]-b[1])
def Multicomplejos(a,b):
    return((a[0]*b[0] - a[1]*b[1]) , (a[1]*b[0] + a[0]*b[1]))
def Cocientecomplejos(a,b):
    return ((a[0]*b[0] + a[1]*b[1])/(b[0]**2 + b[1]**2) , (b[0]*a[1]-a[0]*b[1])/(b[0]**2 + b[1]**2))
def conjugado(a):
    return(a[0] ,(a[1] * -1 ))
def modulo(a):
    return ( a[0]**2 + a[1]**2 )** 1/2 
def cart_a_Polares(a):
    angulo = round(math.atan2(a[1],a[0]),2)
    return((a[0]**2 + a[1]**2 )** 1/2 , angulo)
def fase(a):
    angulo = round(math.atan2(a[1],a[0]),2)
    return (angulo)
def prettyprintings(c):
    print(c[0] , "+" , c[1], "i")
def prettyprintingsPolares(c):
    print("(",c[0], "," ,c[1],")")
def prettyprintingsmodulo(c):
    print("El modulo es:",c)
def prettyprintingsFase(c):
    print("La fase es:",c)
def pretypreting(m):
    for i in range(len(m)):
        for j in range(len(m[0])):
            print(m[i][j], end = "")
        print()
def producto_vetores(v1,v2):
    valor_esperado = 0
    for i in range(len(v1)):
        for j in range(len(v2[0])):
            x = Multicomplejos(v2[j],v1[j])
            valor_esperado = valor_esperado + round(x[0]+ x[1],1)
        break
    return valor_esperado
def hacer_unitaria(m,valor):
    unitaria = [[[0,0] for j in range(len(m))]for i in range(len(m[0]))]
    for i in range(len(unitaria)):
        for j in range(len(unitaria[0])):
            if i == j:
                unitaria[i][i][0] = valor
    return(unitaria)
def resta_matrices(m1,m2):
    matrizResta = [[[0,0] for j in range(len(m1))] for i in range(len(m1[0]))]
    for i in range(len(m1)):
        for j in range(len(m1[0])):
            matrizResta[i][j][0] = m1[i][j][0] - m2[i][j][0]
            matrizResta[i][j][1] = m1[i][j][1] - m2[i][j][1]
    return matrizResta
def prettyprintingsVectores(vector,parametro):
    if parametro == 1:
        print("El resultado es:")
        for i in range(len(vector)):
            for j in range(1):
                print("{0:.1f}".format(vector[i][j][0]),"+","(","{0:.1f}".format(vector[i][j][1]),")","i")
    elif parametro == 3:
        print("El resultado es:")
        for i in range(len(vector)):
            for j in range(1):
                print("{0:.1f}".format(vector[i][0]),"+","(","{0:.1f}".format(vector[i][1]),")","i")
    else:
        print("El resultado es:")
        print(*vector)
def prettyprintingsMatrices(matriz):
    print("El resultado es:")
    for i in range(len(matriz)):
        for j in range(len(matriz)):
            print("{0:.1f}".format(matriz[i][j][0]),"+","(","{0:.1f}".format(matriz[i][j][1]),")","i", end = "  ")
        print()
def prettyprintingsTensor(matriz,n):
    cont = 0
    for i in range(len(matriz)):
        for j in range(1):
            print("{0:.1f}".format(matriz[i][0]),"+","(","{0:.1f}".format(matriz[i][1]),")","i", end = " ")
            cont += 1
            if cont == n:
                print()
                cont = 0
def suma_Vectores(vector1,vector2,filas1):
    vectorSuma = [[[0,0] for j in range(1)] for i in range(filas1)]
    for i in range(filas1):
        for j in range(1):
            vectorSuma[i][j][0] = vector1[i][0] + vector2[i][0]
            vectorSuma[i][j][1] = vector1[i][1] + vector2[i][1]
    return vectorSuma 
def inversa(vector1):
    vectorInver = [[[0,0] for j in range(1)] for i in range(len(vector1))]
    for i in range(len(vector1)):
        for j in range(1):
            vectorInver[i][j][0] = vector1[i][0] * (-1)
            vectorInver[i][j][1] = vector1[i][1] * (-1)
    return vectorInver 
def producto_por_escalar(vector,escalar):
    vectorEsca = [[[0,0] for j in range(1)] for i in range(len(vector))]
    for i in range(len(vector)):
        for j in range(1):
            x = Multicomplejos(vector[i],escalar)
            vectorEsca[i][j][0] = x[0]
            vectorEsca[i][j][1] = x[1]
    return vectorEsca 
def suma_Matrices(matriz1,matriz2):
    matrizSuma = [[[0,0] for j in range(len(matriz1))] for i in range(len(matriz1))]
    for i in range(len(matriz1)):
        for j in range(len(matriz1)):
            matrizSuma[i][j][0] = matriz1[i][j][0] + matriz2[i][j][0]
            matrizSuma[i][j][1] = matriz1[i][j][1] + matriz2[i][j][1]
    return matrizSuma
def Inversa_Matrices(matriz1):
    matrizInversa = [[[0,0] for j in range(len(matriz1))] for i in range(len(matriz1))]
    for i in range(len(matriz1)):
        for j in range(len(matriz1)):
            matrizInversa[i][j][0] = matriz1[i][j][0] * -1
            matrizInversa[i][j][1] = matriz1[i][j][1] * -1
    return matrizInversa 
def matriz_Escalar(matriz1,escalar):
    matrizEscalar = [[[0,0] for j in range(len(matriz1))] for i in range(len(matriz1))]
    for i in range(len(matriz1)):
        for j in range(len(matriz1)):
            x = Multicomplejos(matriz1[i][j],escalar)
            matrizEscalar[i][j][0] = x[0]
            matrizEscalar[i][j][1] = x[1]
    return matrizEscalar
def matriz_Transpuesta(matriz1):
    filas = len(matriz1)
    columnas = len(matriz1[0])
    matrizTranspuesta = [[[0,0] for j in range(filas)]for i in range(columnas)]
    for i in range(columnas):
        for j in range(filas):
            matrizTranspuesta[i][j] = matriz1[j][i]
    return matrizTranspuesta
def matriz_Conjugada(matriz1,valor):
    if valor == 1:
        matriz_Conjugada = [[[0,0] for j in range(len(matriz1))] for i in range(len(matriz1))]
        for i in range(len(matriz1)):
            for j in range(len(matriz1)):
                    matriz_Conjugada[i][j][0] = matriz1[i][j][0]
                    matriz_Conjugada[i][j][1] = matriz1[i][j][1] * -1
        return matriz_Conjugada
    else:
        vectorEsca = [[[0,0] for j in range(1)] for i in range(len(matriz1))]
        for i in range(len(matriz1)):
            for j in range(1):
                vectorEsca[i][j][0] = matriz1[i][0] 
                vectorEsca[i][j][1] = matriz1[i][1] * -1
        return vectorEsca 
def matriz_Producto(m1,m2):
    matrizProducto = [[[0,0] for j in range(len(m1))] for i in range(len(m1))]
    for i in range(len(m1)):
        for j in range(len(m1)):
            for k in range(len(m1[0])):
                matrizProducto[i][j][0] = matrizProducto[i][j][0] + (m1[i][k][0] * m2[k][j][0] - m1[i][k][1]*m2[k][j][1])
                matrizProducto[i][j][1] = matrizProducto[i][j][1] + (m1[i][k][1] * m2[k][j][0] + m1[i][k][0]*m2[k][j][1])
    return matrizProducto
def matriz_sobre_Vector(m1,v1):
    cont = 1
    matriz_Vector = [[[0,0] for j in range(1)] for i in range(len(m1))]
    for i in range(len(m1)):
        for j in range(len(v1[0])):
            for k in range(len(m1[0])):
                matriz_Vector[i][j][0] = matriz_Vector[i][j][0] + (m1[i][k][0] * v1[k][j] - m1[i][k][1]*v1[k][j+cont])
                matriz_Vector[i][j][1] = matriz_Vector[i][j][1] + (m1[i][k][1] * v1[k][j] + m1[i][k][0]*v1[k][j+cont])
            break
    return matriz_Vector
def productoI_Interno(v1,v2):
    cont = [0,0]
    lista = []
    v22 = [[(v2[j][0],v2[j][1]) for j in range(len(v2))] for i in range(1)]
    for i in range(len(v1)):
        for j in range(len(v1)):
            primera = Multicomplejos(v1[j][i],v2[j])
            lista.append(primera)
        break
    for i in range(len(lista)):
        cont = sumComplejos(cont,lista[i])
    return cont
def ditancia_Vectores(v1,v2):
    resta = []
    vector = []
    cont = [0,0]
    for i in range(len(v1)):
        for j in range(1):
            x = Restacomplejos(v1[i],v2[i])
            resta.append(x)
    v3 = list(matriz_Conjugada(resta,2))
    for i in range(len(v1)):
        for j in range(len(v1)):
            primera = Multicomplejos(v3[j][i],resta[j])
            vector.append(primera)
        break
    for i in range(len(vector)):
        cont = sumComplejos(cont,vector[i])
    return cont
def matriz_Unitaria(m1,m2):
    bandera = True
    m1 = matriz_Transpuesta(m1)
    m1 = matriz_Conjugada(m1,1)
    unitaria = matriz_Producto(m1,m2)
    lista = []
    lista2 = []
    for i in range(len(unitaria)):
        for j in range(len(unitaria)):
            if i == j:
                lista.append(unitaria[i][j])
            else:
                lista2.append(unitaria[i][j])
    for i in range(len(lista)-1):
        if lista[i] == lista[i+1] and lista[i] == [1,0] and lista[i+1] == [1,0] :
            bandera
        else:
            bandera = False
    for i in range(len(lista2)-1):
        if lista2[i] == lista2[i+1] and lista2[i] == [0,0] and lista2[i+1] ==[0,0]:
            bandera
        else:
            bandera = False
            break
    return unitaria,bandera
def matriz_Hermitiana(m1,m2):
    bandera = True
    m1 = matriz_Transpuesta(m1)
    m2 = matriz_Conjugada(m2,1)
    for i in range(len(m1)):
        for j in range(len(m2)):
            if m1[i][j][0] == m2[i][j][0] and m1[i][j][1] == m2[i][j][1]:
                bandera
            else:
                bandera = False
    return(bandera)
def producto_Tensor(v1,v2,parametro):
    if parametro == 1:
        tensor = []
        for i in range(len(v1)):
            for j in range(len(v1)):
                for m in range(len(v2)):
                    x = Multicomplejos(v1[j],v2[m])
                    tensor.append(x)
            break
        return tensor
    else:
        filas = len(v1)
        columnas = len(v1[0])
        filas2 = len(v2)
        columnas2 = len(v2[0])
        tensor =[[(0,0) for j in range(columnas*columnas2)]for j in range(filas*filas2)]
        for i in range(len(tensor)):
            for j in range(len(tensor[0])):
                tensor[i][j] = Multicomplejos(v1[i//filas2][j//filas],v2[i%filas2][j%filas])
        return tensor
def hacermatriz(m1,n):
    matriz = [[[0,0] for j in range(n)] for i in range(n)]
    cont = 0
    for i in range(len(matriz)):
        for j in range(len(matriz)):
            for m in range(len(matriz)):
                matriz[j][m][0] = m1[cont][0]
                matriz[j][m][1] = m1[cont][1]
                cont += 1
        break
    cont = 0
    return(matriz)
def hacervector(v1,x):
    cont = 0
    vector = [[0,0] for i in range(x)]
    for i in range(len(vector)):
        if cont <= 2:
            cont += 1
            for j in range(len(vector)):
                if cont <= 2:
                    vector[j][i] = v1[j][0][i] 
    return vector
def leer_vectores_Suma(v1,v2):
    if len(v1) == 0 or len(v2) == 0:
        print("No se puede realizar la suma.")
    elif len(v1) == len(v2):
        vectorSuma = suma_Vectores(v1,v2,len(v1))
        prettyprintingsVectores(vectorSuma,1)
        return vectorSuma
    else:
        print("No se puede realizar la suma.") 
def leer_vectores_Inversa(v1):
    if len(v1) <= 0:
        print("No se puede obtener la inversa del vector.")
    else:
        inversaVector = inversa(v1)
        prettyprintingsVectores(inversaVector,1)
        return inversaVector 
def leer_vectores_por_escalar(v1,c):
    if len(v1) <= 0:
        print("No se puede realizar la operacion.")
    else:
        vectorResultante = producto_por_escalar(v1,c)
        prettyprintingsVectores(vectorResultante,1)
        return vectorResultante
def leer_Matrices_Suma(m1,m2):
    if len(m1) == 0 and len(m2) == 0:
        print("No se puede realizar la suma de matrices.")
    elif len(m1) == len(m2):
        matrizSuma = suma_Matrices(m1,m2)
        prettyprintingsMatrices(matrizSuma)
        return matrizSuma
    else:
        print("No se puede realizar la suma de matrices.")
def leer_Matrices_Inversa(m1):
    if len(m1) <= 0:
        print("No se puede realizar la Inversa de la matriz.")
    else:
        matrizInversa = Inversa_Matrices(m1)
        prettyprintingsMatrices(matrizInversa)
        return matrizInversa 
def leer_Matrices_Escalar(m1,c):
    if len(m1) <= 0:
        print("No se puede realizar la operación.")
    else:
        matrizEscalar = matriz_Escalar(m1,c)
        prettyprintingsMatrices(matrizEscalar)
        return matrizEscalar 
def leer_Matrices_Transpuesta(m1,p):
    if len(m1) <= 0:
        print("No se puede realizar la operación.")
    else:
        if p == 2:
            x = len(m1)
            m1 = matriz_Conjugada(m1,2)
            m1 = hacervector(m1,x)
            m1 = matriz_Conjugada(m1,2)
            m1 = matriz_Transpuesta(m1)
            print(m1)
            return m1
        else:
            m1 = matriz_Transpuesta(m1)
            return m1
def leer_Matrices_Conjugada(m1,valor):
    if valor == 1:
        if len(m1) <= 0:
            print("No se puede realizar la operación.")
        else:
            matrizConjugadad = matriz_Conjugada(m1,valor)
            prettyprintingsMatrices(matrizConjugadad)
            return matrizConjugadad
    else:
        if len(m1) <= 0:
            print("No se puede realizar la operación.")
        else:
            matrizConjugadad = matriz_Conjugada(m1,valor)
            prettyprintingsVectores(matrizConjugadad,1)
            return matrizConjugadad
def leer_Matrices_Adjunta(m1,valor):
    if valor == 1:
        if len(m1) <= 0:
            print("No se puede realizar la operación.")
        else:
            matrizAdjunta = matriz_Transpuesta(m1)
            matrizAdjunta = matriz_Conjugada(matrizAdjunta,valor)
            prettyprintingsMatrices(matrizAdjunta)
            return matrizAdjunta
    else:
        if len(m1) <= 0:
            print("No se puede realizar la operación.")
        else:
            vectorAdjunta = matriz_Conjugada(m1,valor)
            prettyprintingsVectores(vectorAdjunta,valor)
            return vectorAdjunta 
def leer_Matrices_producto(m1,m2):
    if len(m1) == 0 or len(m2) == 0:
        print("No se puede ralizar la operación.")
    elif len(m1) != len(m2):
        print("No se puede ralizar la operación.")
    else:
        matrizProducto = matriz_Producto(m1,m2)
        prettyprintingsMatrices(matrizProducto)
        return matrizProducto 
def leer_accion_matriz_vector(m1,v1):
    if len(m1) == 0 or len(v1) == 0:
        print("No se puede ralizar la operación.")
    elif len(m1[0]) != len(v1):
        print("No se puede ralizar la operación.")
    else:
        matriz_sobreVector = matriz_sobre_Vector(m1,v1)
        prettyprintingsVectores(matriz_sobreVector,1)
        return matriz_sobreVector 
def leer_Producto_Interno(v1,v2):
    if len(v1) == 0 or len(v2) == 0:
        print("No se puede ralizar la operación.")
    elif len(v1) != len(v2):
        print("No se puede ralizar la operación.")
    else:
        v1 = matriz_Conjugada(v1,2)
        productoInterno = productoI_Interno(v1,v2)
        print(productoInterno[0], "+", "(",productoInterno[1],")i")
        print(productoInterno)
        return productoInterno
def leer_vector_norma(v1): 
    if len(v1) == 0:
        print("No se puede ralizar la operación.")
    else:
        print("La norma del vector es:")
        v2 = list(v1)
        v1 = matriz_Conjugada(v1,2)
        productoInterno = productoI_Interno(v1,v2)
        productoInterno = (productoInterno[0]+productoInterno[1]) ** (1/2)
        productoInterno = round(productoInterno,2)
        print(productoInterno)
        return productoInterno
def leer_distancia_vectores(v1,v2):
    if len(v1) <= 0 and len(v2) <= 0:
        print("No se puede ralizar la operación.")
    elif len(v1) != len(v2):
        print("No se puede ralizar la operación.")
    else:
        print("La distancia es:")
        ditanciaVectores = ditancia_Vectores(v1,v2)
        ditanciaVectores = (ditanciaVectores[0]+ditanciaVectores[1]) ** (1/2)
        ditanciaVectores = round(ditanciaVectores,2)
        print(ditanciaVectores)
        return ditanciaVectores

def leer_matriz_Unitaria(m1): 
    if len(m1) == 0:
        print("No se puede ralizar la operación.")
    else:
        m2 = m1
        matrizUnitaria,bandera = matriz_Unitaria(m1,m2)
        if bandera:
            print("Si es unitaria la matriz.")
            prettyprintingsMatrices(matrizUnitaria)
            return matrizUnitaria
        else:
            print("No es unitaria la matriz.")
            prettyprintingsMatrices(matrizUnitaria) 
def leer_matriz_Hermitiana(m1): 
    if len(m1) == 0:
        print("No se puede ralizar la operación.")
    else:
        m2 = m1
        bandera = matriz_Hermitiana(m1,m2)
        if bandera:
            print("Si es Hermitiana la matriz.")
            prettyprintingsMatrices(m1)
            return m1
        else:
            print("No es Hermitiana la matriz.")
            prettyprintingsMatrices(m1)
def leer_producto_Tensor_Vector(v1,v2):
    if len(v1) == 0 or len(v2) == 0:
        print("No se puede ralizar la operación.")
    else:
        print("El producto tensor entre los vectores es:")
        productoTensor = producto_Tensor(v1,v2,1)
        prettyprintingsVectores(productoTensor,3)
        return productoTensor
def leer_producto_Tensor_Matrices(m1,m2):
    if len(m1) == 0 or len(m2) == 0:
        print("No se puede ralizar la operación.")
    else:
        print("El producto tensor es:")
        n = len(m1[0]) * len(m2[0])
        productoTensor = producto_Tensor(m1,m2,2)
        prettyprintingsMatrices(productoTensor)
        return productoTensor
def simulacion():
    x = [[(0,0),(1,0)],[(1,0),(0,0)]]
    h = [[((1/2**(1/2)),0),((1/2**(1/2)),0)],[((1/2**(1/2)),0),((-1/2**(1/2)),0)]]
    o = [(1,0),(0,0)]
    n = len(x[0]) * len(h[0])
    tensor_o = producto_Tensor(o,o,1)
    m1 = producto_Tensor(x,h,2)
    m2 = producto_Tensor(h,h,2)
    matriz1 = hacermatriz(m1,n)
    matriz2 = hacermatriz(m2,n)
    gamma1 = matriz_Producto(matriz1,matriz2)
    gammafinal = matriz_sobre_Vector(gamma1,tensor_o)
    prettyprintingsVectores(gammafinal,1)
def posibilidad_posicion(vector,posicion):
    posicion_vector = [[None,None]]
    posicion_vector[0][0] = vector[posicion][0]
    posicion_vector[0][1] = vector[posicion][1]
    posicion_vector = (posicion_vector[0][0]**2+posicion_vector[0][1]**2)
    v2 = list(vector)
    vector = matriz_Conjugada(vector,2)
    norma = productoI_Interno(vector,v2)
    norma = (norma[0]+norma[1]) ** (1/2)
    probabilidad = posicion_vector/norma**2
    probabilidad = round(probabilidad,6)
    return probabilidad
def amplitud_de_transicion(v1,v2):
    v1,v2 = v2,v1
    v11 = list(v1)
    v1 = matriz_Conjugada(v1,2)
    norma = productoI_Interno(v1,v11)
    norma = (norma[0]+norma[1]) ** (1/2)
    v22 = list(v2)
    x = len(v22)
    v2 = matriz_Conjugada(v2,2)
    norma2 = productoI_Interno(v2,v22)
    norma2 = (norma2[0]+norma2[1]) ** (1/2)
    v2 = hacervector(v2,x)
    v2 = matriz_Conjugada(v2,2)
    v1 = matriz_Transpuesta(v1)
    producto = matriz_Producto(v1,v2)
    noma_Total = norma * norma2
    for i in range(len(producto[0][0])):
        for j in range(1):
            producto[0][0][i] = round(producto[0][0][i] / noma_Total,2)
    return producto
def varianza(matriz,vetor):
    x = matriz_Hermitiana(matriz,matriz)
    if x:
        print("Es hermitiana.")
    else:
        print("No es hermitiana.")
def mirar_hermitiana(matriz):
    x = matriz_Hermitiana(matriz,matriz)
    if x:
        return True
    else:
        return False
def valorEsperado(v,m):
    v2 = matriz_sobre_Vector(m,v)
    v2 = hacervector(v2,len(v))
    v2 = matriz_Conjugada(v2,2)
    v2 = hacervector(v2,len(v))
    valor_esperado = producto_vetores(v2,v)
    return valor_esperado
def varianza(ket, matriz):
    valor_esperado = valorEsperado(ket,matriz)
    matrizUnitaria = hacer_unitaria(matriz,valor_esperado)
    resta = resta_matrices(matriz,matrizUnitaria)
    produto = matriz_Producto(resta,resta)
    produto1 = hacervector(matriz_sobre_Vector(produto,ket),len(ket))
    conjugada = hacervector(matriz_Conjugada(ket,2),len(ket))
    varian = producto_vetores(conjugada,produto1)
    return(varian)
def valores_esperados(matriz,parametro):
    valor = []
    valores, vectores = np.linalg.eig(matriz)
    vecto = [[] for i in range(len(vectores))]
    for i in range(len(valores)):
        valor.append(round(valores[i],1))
    for i in range(len(vectores)):
        for j in range(len(vectores)):
            vecto[i].append(vectores[i][j])
    if parametro == 1:
        return valor
    else:
        return vecto
def probabilidad(vector_estado1,vector_estado2,matriz,valor):
    if matriz == [[0,-1j],[1j,0]]:
        vector1 = [[0,1],[1,0]]
        vector2 = [[0,-1],[1,0]]
    else:
        vectores_propios = valores_esperados(matriz,2)
        vector1 = [[0,0] for i in range(len(vectores_propios[0]))]
        vector2 = [[0,0] for i in range(len(vectores_propios[0]))]
        if vectores_propios[0][0] != complex:
            vector1[0][0] = vectores_propios[0][0]
        else:
            vector1[0][1] = vectores_propios[0][0]
        if vectores_propios[0][1] != complex:
            vector1[1][0] = vectores_propios[0][1]
        else:
            vector1[1][1] = vectores_propios[0][1]
        #2
        if vectores_propios[1][0] != complex:
            vector2[0][0] = vectores_propios[1][0]
        else:
            vector2[0][1] = vectores_propios[1][0]
        if vectores_propios[1][1] != complex:
            vector2[1][0] = vectores_propios[1][1]
        else:
            vector2[1][1] = vectores_propios[1][1]
    amplitud1 = amplitud_de_transicion(vector_estado1,vector1)
    amplitud2 = amplitud_de_transicion(vector_estado1,vector2)
    amplitud3 = amplitud_de_transicion(vector_estado2,vector1)
    amplitud4 = amplitud_de_transicion(vector_estado2,vector2)
    if valor == 1:
        return amplitud1
    elif valor == 2:
        return amplitud2
    elif valor == 3:
        return amplitud3
    elif valor == 4:
        return amplitud4
def comprobar_producto(m1,m2):
    m3 = m1
    m4 = m2
    bandera1 = matriz_Hermitiana(m1,m3)
    bandera2 = matriz_Hermitiana(m2,m4)
    if bandera1 and bandera2:
        producto = matriz_Producto(m1,m2)
        m5 = producto
        bandera3 = matriz_Hermitiana(producto,m5)
        if bandera3:
            return bandera3
        else:
            return bandera3
    else:
        return False
def probabilidad_3_clic(m,v):
    x = len(v)
    for i in range(3):
        vector = matriz_sobre_Vector(m,v)
        vector = hacervector(vector,x)
        v = vector
    probabilidad = round(v[2][0] ** 2 + v[2][1] ** 2,4)
    return probabilidad
