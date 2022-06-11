#Importar las librerías por utilizar
import os
import cv2
import numpy as np
import time
from skimage import io
import playsound as ps
from skimage.transform import resize
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from skimage.color import rgb2hsv
from skimage import exposure
from skimage.morphology import disk
from skimage.morphology import (closing)
from skimage.measure import label, regionprops, regionprops_table
from skimage import exposure


#Obtener la direccion de las imagenes
path = os.getcwd()

#Lectura de las imagenes de prueba
folder = 'MUESTRAS'
folder_path = os.path.join(path, folder)
files = os.listdir(folder_path)
lista_fotos = []
for file in files:
    file_path = os.path.join(folder_path, file)
    foto = io.imread(file_path)
    lista_fotos.append(foto)

#PARAMETROS

#Ejecutar para informe
INF = False

#Recortar la imagen(%)
Y_MIN = 0.5
Y_MAX = 1
X_MIN = 0
X_MAX = 1

#Escalado
X_PX = 320
Y_PX = 360

#numero de colores incluyendo el fondo
N_COLORS = 5


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

def stretch_hsv(img_hsv,channel):
    p2, p98 = np.percentile(img_hsv[:,:,channel], (2, 98))
    stretched = exposure.rescale_intensity(img_hsv[:,:,channel], in_range=(p2, p98))
    return stretched

def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    return codebook[labels].reshape(w, h, -1)

def img_to_5_colors(img):
    # Copia de la imagen por cuantificar
    img_5_colors = img

    # Convert to floats instead of the default 8 bits integer coding. Dividing by
    # 255 is important so that plt.imshow behaves works well on float data (need to
    # be in the range [0-1])
    img_5_colors = np.array(img_5_colors, dtype=np.float64)

    # Load Image and transform to a 2D numpy array.
    w, h, d = tuple(img_5_colors.shape)
    assert d == 3
    image_array = np.reshape(img_5_colors, (w * h, d))
    image_array_sample = shuffle(image_array, random_state=0, n_samples=1_000)
    kmeans = KMeans(n_clusters=N_COLORS, random_state=0).fit(image_array_sample)
    labels = kmeans.predict(image_array)
    colores = kmeans.cluster_centers_
    #coloresh = rgb2hsv(colores)[:,0]
    #coloress = rgb2hsv(colores)[:,1]
    #coloresv = rgb2hsv(colores)[:,2]
    coloresh = colores[:,0]
    coloress = colores[:,1]
    coloresv = colores[:,2]
    #print(coloresh)
    #print(coloress)
    #print(coloresv)
    #pru = recreate_image(colores, labels, w, h)
    #plt.imshow(pru)
    for i in range(N_COLORS):
        if not (0.07 < coloresh[i] < 0.19 and coloress[i] > 0.219 and coloresv[i] > 0.15):
            colores[i] = [0, 0, 0]
        new_img = recreate_image(colores, labels, w, h)
    #plt.imshow(new_img)
    new_img = rgb2hsv(new_img)[:,:,2]
    new_img = new_img > 0 #threshold_otsu(new_img)
    footprint = disk(10)
    new_img = closing(new_img,footprint)
    return new_img

def preproc(img):
    resized = edit_image(img)
    img_RGB =rgba2rgb(resized)
    img_HSV =rgb2hsv(img_RGB)
    img_eq = img_HSV
    img_eq[:,:,1] = stretch_hsv(img_HSV,1)
    img_eq[:,:,2] = stretch_hsv(img_HSV,2)
    #plt.imshow(hsv2rgb(img_eq))
    #plt.show()    
    quant = img_to_5_colors(img_HSV)
    return quant

# Funcion de segmentacion
def segmentacion(img):
    segmented = label(img, connectivity= None)
    props = regionprops(segmented)
    for prop in props:
        if prop.area > 10000 and prop.label != 0:
            region_label = prop.label
            region = (segmented == region_label)*1
            centroide = prop.centroid
    return region, centroide

# Tipo de interseccion y esquinas
def intersecciones2(region):
    r_border = region[:,300:320]
    l_border = region[:,0:20]
    u_border = region[0:40,:] # indica posible interseccion cruz o T
    ur_corner = region[0:20,300:320]
    ul_corner = region[0:20,0:20]
    r_path = np.sum(r_border) > 800 #existe camino der
    l_path = np.sum(l_border) > 800 #existe camino izq
    u_path = np.sum(u_border) > 500 #posible camino recto
    empty_corners = (np.sum(ur_corner) < 100) and (np.sum(ul_corner) < 100) #pocos o nulos pixeles en las esquinas
    #es decir la interseccion no queda en el borde para distinguir entre codo e interseccion
    #Intersecciones
    cruz = l_path and r_path and u_path
    Te = l_path and r_path and (not u_path)
    i_l = l_path and (not r_path) and u_path
    i_r = (not l_path) and r_path and u_path
    codo_l = l_path and (not r_path) and (not u_path)
    codo_r = (not l_path) and r_path and (not u_path)
    recto = (not l_path) and (not r_path) and u_path
    intersecciones = [cruz, Te, i_l, i_r, codo_l, codo_r, recto]
    interseccion = intersecciones.index(True)
    return interseccion, empty_corners

    
# Toma una instruccion y una interseccion y toma decision
def decision2(inst, interseccion, empty_corners):
    giro = False
    if empty_corners:
        #Cruz
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
        #Te
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
            #Si se inidica continuar y solo se puede girar
            elif inst == "c":
                print("Intruccion no valida \n")
                print("Tome otra direccion \n")
                os.startfile("invalida.mp3")
                time.sleep(5)                
        #Interseccion izq
        elif interseccion == 2:
            if inst == "l":
                print("Gire a la izq \n Girando \n")
                os.startfile("izquierda.mp3")
                time.sleep(5)
                giro = True
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
        #Interseccion der
        elif interseccion == 3:
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
        #Codo izq/der
        elif interseccion == 4 or interseccion == 5:
            print("Codo detectado \n Continue \n")
            os.startfile("adelante.mp3")
            time.sleep(5)
        #Recto
        elif interseccion == 6:
            print("Continue \n")
    else:
        print("Continue \n")
    return giro

def correccion(centroide):
    x_c = centroide[1]
    #Se determina la distancia en x del centroide al centro de la imagen y 
    #si el desvio es muy alto indica que se debe hacer una correccion
    desviacion = abs(x_c - 160)
    if desviacion >= 20:
        print("pip, pip, pip")
        os.startfile("pipipip.mp3")

folder = 'SECUENCIA'
folder_path = os.path.join(path, folder)
files = os.listdir(folder_path)
lista_fotos = []
for file in files:
    file_path = os.path.join(folder_path, file)
    foto = io.imread(file_path)
    lista_fotos.append(foto)

def main():
    cam = cv2.VideoCapture(0)
    instrucciones = ["l","l","r","r"]
    i = 0
    while i < len(instrucciones):
        ret, frame = cam.read()
        img_name = "IMG_ACTUAL.png"
        cv2.imwrite(img_name, frame)
        file_path_act = os.path.join(path, img_name)
        foto = io.imread(file_path_act)
        quant = preproc(foto)
        region, centroide = segmentacion(quant)
        intersec, esquinas = intersecciones2(region)
        if intersec == 6:
            correccion(centroide)
        giro = decision2(instrucciones[i], intersec, esquinas)
        if giro:
            i +=1
        if i == len(instrucciones):
            break
    print("Se ejecutaron todas las instrucciones")
    os.startfile("finalizado.mp3")
    cam.release()

if __name__ == "__main__": main()