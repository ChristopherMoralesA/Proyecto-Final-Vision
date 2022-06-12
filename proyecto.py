#----------------------LIBRERIAS----------------------
import os
import cv2
import time
import numpy as np
from skimage import io
from skimage import exposure
from sklearn.utils import shuffle
from skimage.color import rgb2hsv
from sklearn.cluster import KMeans
from skimage.morphology import disk
from skimage.transform import resize
from skimage.morphology import closing
from skimage.measure import label, regionprops



#----------------------PARAMETROS----------------------
#Obtener la direccion de las imagenes
path = os.getcwd()

# Recortar la imagen(%)
Y_MIN = 0.3
Y_MAX = 1
X_MIN = 0.15
X_MAX = 0.85

# Escalado
X_PX = 320
Y_PX = 360

# Numero de colores incluyendo el fondo
N_COLORS = 5



# ----------------------FUINCIONES----------------------
# Escala y Recorta la imagen a 360x320
def edit_image(image):
    y,x = image.shape[:2]
    cropped = image[int(y*Y_MIN):int(y*Y_MAX),int(x*X_MIN):int(x*X_MAX)]
    resized = resize(cropped,(Y_PX,X_PX),preserve_range=True).astype(int)
    return resized


# Convierte la imagen RGBA a RGB
def rgba2rgb( rgba, background=(255,255,255) ):
    row, col, ch = rgba.shape
    if ch == 3:
        return rgba
    assert ch == 4, 'RGBA image has 4 channels.'
    rgb = np.zeros( (row, col, 3), dtype='float32' )
    r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]
    a = np.asarray( a, dtype='float32' ) / 255.0
    R, G, B = background
    rgb[:,:,0] = r * a + (1.0 - a) * R
    rgb[:,:,1] = g * a + (1.0 - a) * G
    rgb[:,:,2] = b * a + (1.0 - a) * B
    return np.asarray( rgb, dtype='uint8' )


# Realiza stetching al canal de hsv indicado
def stretch_hsv(img_hsv,channel):
    p2, p98 = np.percentile(img_hsv[:,:,channel], (2, 98))
    # Realiza el stretching en los 96 percentiles del medio, entre 2 y 98
    stretched = exposure.rescale_intensity(img_hsv[:,:,channel], in_range=(p2, p98))
    return stretched


# Recrea la imagen
def recreate_image(codebook, labels, w, h):
    return codebook[labels].reshape(w, h, -1)


# Cuantifica la imagen en 5 colores, binariza dejando unicamente regiones de color
# amarillo y une regiones que estan separadas pero son una misma
def img_to_5_colors(img):
    # Copia de la imagen por cuantificar
    img_5_colors = img
    # Convierte a floats en lugar de los enteros de 8 bits por defecto
    img_5_colors = np.array(img_5_colors, dtype=np.float64)
    # Carga la imagen como un arrglo 2D de numpy.
    w, h, d = tuple(img_5_colors.shape)
    assert d == 3
    image_array = np.reshape(img_5_colors, (w * h, d))
    image_array_sample = shuffle(image_array, random_state=0, n_samples=1_000)
    kmeans = KMeans(n_clusters=N_COLORS, random_state=0).fit(image_array_sample)
    labels = kmeans.predict(image_array)
    colores = kmeans.cluster_centers_
    coloresh = colores[:,0]
    coloress = colores[:,1]
    coloresv = colores[:,2]
    # Binariza la imagen utilizando los canales hsv para identificar el color amarillo 
    # de interes y poniendo todo lo demas en negro
    for i in range(N_COLORS):
        if not (0.07 < coloresh[i] < 0.19 and coloress[i] > 0.219 and coloresv[i] > 0.15):
            colores[i] = [0, 0, 0]
        new_img = recreate_image(colores, labels, w, h)
    new_img = rgb2hsv(new_img)[:,:,2]
    new_img = new_img > 0 #threshold_otsu(new_img)
    # Une regiones
    footprint = disk(10)
    new_img = closing(new_img,footprint)
    return new_img


# Funcion de preprocesado de la imagen, ejecuta funciones definidas aneriormente
def preproc(img):
    resized = edit_image(img)
    img_RGB =rgba2rgb(resized)
    img_HSV =rgb2hsv(img_RGB)
    img_eq = img_HSV
    img_eq[:,:,1] = stretch_hsv(img_HSV,1)
    img_eq[:,:,2] = stretch_hsv(img_HSV,2)   
    quant = img_to_5_colors(img_HSV)
    return quant


# Funcion de segmentacion, segmenta la imagen en regiones y guarda la region mas
# grande correspondiente al camino, retornando tambien el centroide de dicha region
def segmentacion(img):
    segmented = label(img, connectivity= None)
    props = regionprops(segmented)
    region1 = (segmented)*0
    centroide = 160
    for prop in props:
        if prop.area > 10000 and prop.label != 0:
            region_label = prop.label
            region1 = (segmented == region_label)*1
            centroide = prop.centroid
    return region1, centroide


# Identifica el tipo de interseccion y esquinas
def intersecciones(region):
    # Aisla las regiones de interes
    r_border = region[:,300:320]
    l_border = region[:,0:20]
    u_border = region[0:40,:] # indica posible interseccion cruz o T
    ur_corner = region[0:20,300:320]
    ul_corner = region[0:20,0:20]
    r_path = np.sum(r_border) > 800 #existe camino hacia la der
    l_path = np.sum(l_border) > 800 #existe camino hacia la izq
    u_path = np.sum(u_border) > 500 #posible camino recto
    empty_corners = (np.sum(ur_corner) < 100) and (np.sum(ul_corner) < 100) #pocos o nulos pixeles en las esquinas
    # es decir la interseccion no queda en el borde para distinguir entre codo e interseccion
    # Intersecciones posibles
    cruz = l_path and r_path and u_path
    Te = l_path and r_path and (not u_path)
    i_l = l_path and (not r_path) and u_path
    i_r = (not l_path) and r_path and u_path
    codo_l = l_path and (not r_path) and (not u_path)
    codo_r = (not l_path) and r_path and (not u_path)
    recto = (not l_path) and (not r_path) and u_path
    # Matriz de posibles intersecciones
    intersecciones = [cruz, Te, i_l, i_r, codo_l, codo_r, recto,True]
    interseccion = intersecciones.index(True)
    # Retorna el indice de la interseccion correspondiente con la imagen
    return interseccion, empty_corners

    
# Toma una instruccion y una interseccion y toma decision
def decision(inst, interseccion, empty_corners):
    # Bandera que indica que se consumio una instruccion
    giro = False
    # Toma la decision unicamente si no hay esquinas superiores,
    # de forma que se eviten confusiones
    if empty_corners:
        # Si la interseccion es en cruz
        if interseccion == 0:
            if inst == "l":
                print("Gire a la izq \n Girando \n")
                os.startfile("izquierda.mp3")
                time.sleep(5)
                giro = True
            elif inst == "r":
                print("Gire a la der \n Girando \n")
                os.startfile("derecha.mp3")                
                time.sleep(5)
                giro = True
            elif inst == "c":
                print("Continue directo \n")
                os.startfile("adelante.mp3")
                time.sleep(5)
                giro = True
        # Si la interseccion es en T
        elif interseccion == 1:
            if inst == "l":
                print("Gire a la izq \n Girando \n")
                os.startfile("izquierda.mp3")
                time.sleep(5)
                giro = True
            elif inst == "r":
                print("Gire a la der \n Girando \n")
                os.startfile("derecha.mp3")
                time.sleep(5)
                giro = True
            # Si se inidica continuar y solo se puede girar
            elif inst == "c":
                print("Intruccion no valida \n")
                print("Tome otra direccion \n")
                os.startfile("invalida.mp3")
                time.sleep(5)                
        # Si la interseccion es unicamente a la izquierda
        elif interseccion == 2:
            if inst == "l":
                print("Gire a la izq \n Girando \n")
                os.startfile("izquierda.mp3")
                time.sleep(5)
                giro = True
            # Si decide ir a la derecha cuando no hay camino
            elif inst == "r":
                print("Intruccion no valida \n")
                print("Continue directo \n")
                os.startfile("adelante.mp3")
                time.sleep(5)
            elif inst == "c":
                print("Continue directo \n")
                os.startfile("adelante.mp3")
                time.sleep(5)
                giro = True
        # Si la interseccion es unicamente a la derecha
        elif interseccion == 3:
            # Si decide ir a la izquierda cuando no hay camino
            if inst == "l":
                print("Intruccion no valida \n")
                print("Continue directo \n")
                os.startfile("adelante.mp3")
                time.sleep(5)            
            elif inst == "r":
                print("Gire a la der \n Girando \n")
                os.startfile("derecha.mp3")
                time.sleep(5)
                giro = True
            elif inst == "c":
                print("Continue directo \n")
                os.startfile("adelante.mp3")
                time.sleep(5)
                giro = True        
        # Si hay un codo izq/der
        elif interseccion == 4 or interseccion == 5:
            print("Continue \n")
            time.sleep(5)
        # Si el camino es recto
        elif interseccion == 6:
            print("Continue \n") 
        #Si no hay intersecciones de ningun tipo
        else: 
            print("Continue \n")
    # En caso de que existan esquinas no hace evaluacion y continua esperando 
    # a que se pueda tomar una decicion correctamente
    else:
        print("Continue \n")
    return giro


# Funcion que determina la necesidad de correccion del recorrido
def correccion(centroide):
    x_c = centroide[1]
    # Se determina la distancia en x del centroide al centro de la imagen y 
    # si el desvio es muy alto indica que se debe hacer una correccion
    desviacion = abs(x_c - 160)
    if desviacion >= 50:
        print("pip, pip, pip")
        os.startfile("pipipip.mp3")


# Funcion principal que ejecuta la secuencia de instrucciones
def main():
    # Reserva la camara conectada
    cam = cv2.VideoCapture(1)
    # Lista con las instrucciones a utilizar
    instrucciones = ["c","c","l"]
    i = 0
    # Se ejecuta hasta que se realicen todas las instrucciones
    while i < len(instrucciones):
        # Captura una imagen
        ret, frame = cam.read()
        img_name = "IMG_ACTUAL_{}.png".format(i)
        # Guarda la imagen
        cv2.imwrite(img_name, frame)
        file_path_act = os.path.join(path, img_name)
        # Accede a la imagen capturada
        foto = io.imread(file_path_act)
        # Realiza el preprocesado de la imagen
        quant = preproc(foto)
        # Realiza la segmentacion y obtencion del centroide
        region2, centroide = segmentacion(quant)
        # Determina la interseccion
        intersec, esquinas = intersecciones(region2)
        # Realiza la revision de la correccion
        if intersec == 6:
            correccion(centroide)
        # Determina la direccion y si se realizo una instruccion
        giro = decision(instrucciones[i], intersec, esquinas)
        if giro:
            i +=1
        if i == len(instrucciones):
            break
    # Indica la finalizacion de la ejecucion
    print("Se ejecutaron todas las instrucciones")
    os.startfile("finalizado.mp3")
    cam.release()


if __name__ == "__main__": main()