from numpy import linspace, array, concatenate, append

from pandas import DataFrame, read_csv

from sklearn.cluster import DBSCAN

import rasterio

from numba import njit
from numba.typed import List


@njit
def mesureTreeHeight(x,y,z,matrix,geoTrans):
    '''
    Funcion que proyecta las coordenadas correspondientes a cada punto de un archivo las sobre un dtm, con la finalidad de
    trabajar con la altura absoluta de cada punto. Esto con la intencion de erradicar las posibles diferencias en alturas 
    provocadas por la topografia. Notese que la funcion esta optimizada con Numba, por lo que resulta eficiente para grandes 
    volumes de datos.
    
    @param x: coordenada en x de cada punto del archivo .las
    @param y: coordenada en y de cada punto del archivo .las
    @param z: elevacion de cada punto del archivo .las
    @param matrix: dtm como numpy array
    @param geoTrans: vector de geotransformacion segun rasterio
    
    @return h: lista con todas las alturas absolutas de cada punto del archivo .las

    '''
    
    h = List()
    
    cx= geoTrans[0]/2 + geoTrans[2]
    cy= geoTrans[4]/2  + geoTrans[-4]
    
    for i,j,k in zip(x,y,z):
        icol = round((i-cx)/geoTrans[0])
        irow = round((j-cy)/geoTrans[4])
        h.append(k-rmatriz[irow,icol])
        
    return h 


def inputDBscan(df,x,y):
    '''
    Función que crea una estructura de entrada entendible para DBSCAN de scikit-learn a partir un dataframe de pandas con 
    las coordenadas (por columna) de un conjunto de puntos.
    
    @param df: dataframe con las coordenadas
    @param x: cadena correspondiente al nombre de la columna que contiene los valores de la componente x
    @param y: cadena correspondiente al nombre de la columna que contiene los valores de la componente y
    
    @return lista entendible como input para scikit-learn
    '''
    return [[i,j] for i,j in zip(df[x],df[y])]



def dbscanTrees(data, eps, samp):
    '''
    Función que implementa el algoritmo DBSCAN de scikit-learn sobre un conjunto de puntos.
    
    @param data: lista con el conjunto de puntos a los que se les aplicará el algoritmo
    @param eps: parametro de entrada para DBSCAN (radio de vecinidad)
    @param samp: parametro de entrada para DBSCAN (numero minimo de elementos por cluster)

    @return labels: lista que contiene la etiqueta del cluster asignado para cada punto de entrada
    '''
    
    clustering = DBSCAN(eps = eps, min_samples = samp).fit(data)
    labels = clustering.labels_

    print('Número de troncos: ',len(set(labels)) - (1 if -1 in labels else 0))

    return labels


if __name__ == '__main__':

	raster = rasterio.open("terreno_Yuncos.tif") # abre dtm

	csv=read_csv("Yuncos_arboles_corte.csv",usecols=["X","Y","ELEV"],
            dtype={"x":float,"Y":float,"ELEV":float}) # abre csv con los puntos correspondientes a vegetacion de un archivo .las

	rmatriz=raster.read(1) # lee el dtm como numpy array

	geoTrans = tuple(raster.transform) # obtiene el vector de geotransformacion del dtm

	# convierte las columnas correspondientes a las componentes x,y,z del csv a una estructura de datos
	# optima para numba
	x = List(csv.X)
	y = List(csv.Y)
	z = List(csv.ELEV)

	alturas = mesureTreeHeight(x,y,z,rmatriz,geoTrans) # Obtiene la altura absoluta de cada punto

	csv['alturas']=list(alturas) # se asigna el vector de alturas como una nueva columna en el csv del .las

	corte3=csv[(csv.alturas<1.61) & (csv.alturas>1.39)].copy() # hace un corte transversal para extraer las secciones de los troncos

	corte3['label'] = dbscanTrees(corte3, 1.5, 3) # clasifica cada punto como perteneciente a algun arbol y asigna la etiqueta en el csv cargado

	corte3.to_csv('outputs/output_167dpy.csv',index=False) # guarda
