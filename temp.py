# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from __future__ import print_function
#import potrebnih biblioteka
#%matplotlib inline
import cv2
import numpy as np 
import matplotlib.pyplot as plt 
import collections
from scipy import signal
from keras.models import model_from_json
# keras
from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.optimizers import SGD
from skimage import img_as_ubyte

import matplotlib.pylab as pylab
pylab.rcParams['figure.figsize'] = 16, 12 # za prikaz većih slika i plotova, zakomentarisati ako nije potrebno

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.uint8)
    ret,image_bin = cv2.threshold(image_gs, 174, 255, cv2.THRESH_BINARY)
    return image_bin
def invert(image):
    return 255-image
def display_image(image, color= False):
    if color:
        #plt.figure()
        plt.imshow(image)
    else:
        plt.imshow(image, 'gray')
def dilate(image):
    kernel = np.ones((3,3)) # strukturni element 3x3 blok
    return cv2.dilate(image, kernel, iterations=1)
def erode(image):
    kernel = np.ones((3,3)) # strukturni element 3x3 blok
    return cv2.erode(image, kernel, iterations=1)

def resize_region(region):
    '''Transformisati selektovani region na sliku dimenzija 28x28'''
    return cv2.resize(region,(28,28), interpolation = cv2.INTER_NEAREST)
def select_roi(image_orig, image_bin,plus):
    '''Oznaciti regione od interesa na originalnoj slici. (ROI = regions of interest)
        Za svaki region napraviti posebnu sliku dimenzija 28 x 28. 
        Za označavanje regiona koristiti metodu cv2.boundingRect(contour).
        Kao povratnu vrednost vratiti originalnu sliku na kojoj su obeleženi regioni
        i niz slika koje predstavljaju regione sortirane po rastućoj vrednosti x ose
    '''
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sorted_regions = [] # lista sortiranih regiona po x osi (sa leva na desno)
    regions_array = []
    #print(plus.x1,plus.y1,plus.x2,plus.y2)

    for contour in contours: 
        x,y,w,h = cv2.boundingRect(contour) #koordinate i velicina granicnog pravougaonika
        area = cv2.contourArea(contour)
        razlika=11
        if abs((plus.x1*(plus.y2-(y-razlika))+plus.x2*((y-razlika)-plus.y1)+(x-razlika)*(plus.y1-plus.y2))/2)<140 and x>plus.x1-5 and x<plus.x2+15:
            
            #if h < 50 and h > 10 and w > 5:
            if (((w > 1 and h > 10) or (w>14 and h>5)) and (w<=30 and h<=30)) and area>=40:
            #if (((w > 1 and h > 10) or (w>14 and h>5)) and (w<=30 and h<=30)) and (w>15 or h>15):

            # kopirati [y:y+h+1, x:x+w+1] sa binarne slike i smestiti u novu sliku
            # označiti region pravougaonikom na originalnoj slici (image_orig) sa rectangle funkcijom
                okvir=2
                region = image_bin[y-okvir:y+h+1+okvir,x-okvir:x+w+1+okvir]
                regions_array.append([resize_region(region), (x,y,w,h)])       
                cv2.rectangle(image_orig,(x,y),(x+w,y+h),(0,255,0),2)
    regions_array = sorted(regions_array, key=lambda item: item[1][0])
    sorted_regions = sorted_regions = [region[0] for region in regions_array]
    
    # sortirati sve regione po x osi (sa leva na desno) i smestiti u promenljivu sorted_regions
    return image_orig, sorted_regions
def scale_to_range(image): # skalira elemente slike na opseg od 0 do 1
    ''' Elementi matrice image su vrednosti 0 ili 255. 
        Potrebno je skalirati sve elemente matrica na opseg od 0 do 1
    '''
    return image/255
def matrix_to_vector(image):
    '''Sliku koja je zapravo matrica 28x28 transformisati u vektor sa 784 elementa'''
    return image.flatten()
def prepare_for_ann(regions):
    '''Regioni su matrice dimenzija 28x28 čiji su elementi vrednosti 0 ili 255.
        Potrebno je skalirati elemente regiona na [0,1] i transformisati ga u vektor od 784 elementa '''
    ready_for_ann = []
    for region in regions:
        # skalirati elemente regiona 
        # region sa skaliranim elementima pretvoriti u vektor
        # vektor dodati u listu spremnih regiona
        scale = scale_to_range(region)
        ready_for_ann.append(matrix_to_vector(scale))
        
    return ready_for_ann
def convert_output(alphabet):
    '''Konvertovati alfabet u niz pogodan za obučavanje NM,
        odnosno niz čiji su svi elementi 0 osim elementa čiji je
        indeks jednak indeksu elementa iz alfabeta za koji formiramo niz.
        Primer prvi element iz alfabeta [1,0,0,0,0,0,0,0,0,0],
        za drugi [0,1,0,0,0,0,0,0,0,0] itd..
    '''
    nn_outputs = []
    for index in range(len(alphabet)):
        output = np.zeros(len(alphabet))
        output[index] = 1
        nn_outputs.append(output)
    return np.array(nn_outputs)
def create_ann():
    '''Implementacija veštačke neuronske mreže sa 784 neurona na uloznom sloju,
        128 neurona u skrivenom sloju i 10 neurona na izlazu. Aktivaciona funkcija je sigmoid.
    '''
    ann = Sequential()
    ann.add(Dense(128, input_dim=784, activation='sigmoid'))
    ann.add(Dense(10, activation='sigmoid'))
    return ann
    
def train_ann(ann, X_train, y_train):
    '''Obucavanje vestacke neuronske mreze'''
    X_train = np.array(X_train, np.float32) # dati ulazi
    y_train = np.array(y_train, np.float32) # zeljeni izlazi za date ulaze
   
    # definisanje parametra algoritma za obucavanje
    sgd = SGD(lr=0.01, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)

    # obucavanje neuronske mreze
    ann.fit(X_train, y_train, epochs=2000, batch_size=1, verbose = 0, shuffle=False) 
      
    return ann
def winner(output): # output je vektor sa izlaza neuronske mreze
    '''pronaći i vratiti indeks neurona koji je najviše pobuđen'''
    return max(enumerate(output), key=lambda x: x[1])[0]
def display_result(outputs, alphabet):
    '''za svaki rezultat pronaći indeks pobedničkog
        regiona koji ujedno predstavlja i indeks u alfabetu.
        Dodati karakter iz alfabet u result'''
    result = []
    for output in outputs:
        result.append(alphabet[winner(output)])
    return result

def get_koord(lines):
    x1=lines[0][0][0]
    y1=lines[0][0][1]
    x2=lines[0][0][2]
    y2=lines[0][0][3]
    for line in lines:
        if line[0][0]<x1:
            x1=line[0][0]
        if line[0][1]>y1:
            y1=line[0][1]
        if line[0][2]>x2:
            x2=line[0][2]
        if line[0][3]<y2:
            y2=line[0][3]
            
    return x1,y1,x2,y2
class Tacka:
    def __init__(self, x1,y1,x2,y2):
        self.x1=x1
        self.y1=y1
        self.x2=x2
        self.y2=y2
    
def hist(image):
    height, width = image.shape[0:2]
    x = range(0, 256)
    y = np.zeros(256)
    
    for i in range(0, height):
        for j in range(0, width):
            pixel = image[i, j]
            y[pixel] += 1
    
    return (x, y)
def jedinstven(y,broj_piksela):
    #print(broj_piksela[-1])
    if abs(y[-1]-(broj_piksela[-1]))<7:
        return False
    if abs(y[-1]-(broj_piksela[-2]))<7:
        return False
    return True
# ucitavanje videa
    
json_file = open('model.json', 'r')
model_json = json_file.read()
json_file.close()
ann = model_from_json(model_json)
ann.load_weights("model.h5")

file= open("out.txt","w+")
file.write("RA 8/2015 Dragan Skiljevic\r")
file.write("file	sum\r")


for i in range(0,10):
    cap = cv2.VideoCapture('video/video-'+str(i)+'.avi')
    frame_num = 0
    cap.set(1, frame_num) # indeksiranje frejmova
    
    # analiza videa frejm po frejm
    #while True:
        #frame_num += 1
    ret_val, frame = cap.read()
        # plt.imshow(frame)
        # ako frejm nije zahvacen
        #if not ret_val:
         #   break
    #print(frame_num)
    kernel = np.ones((3, 3)) # strukturni element 3x3 blok
    kerneljedan = np.ones((1, 1))
    kerneldil = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3)) # MORPH_ELIPSE, MORPH_RECT...
    kerneldil[1][1]=0
    kerneldil[0][1]=0
    kerneldil[1][2]=0
    kerneldil1=np.zeros((2,2))
    kerneldil1[1][1]=1
    kernelnas = np.ones((5,5),np.uint8)
    #display_image(frame)
    #plt.figure()
        # dalje se sa frejmom radi kao sa bilo kojom drugom slikom, npr
    pom=frame[:,:,0]
    
    #display_image(pom)
    #plt.figure()
    
    #frame_gray = cv2.cvtColor(pom, cv2.COLOR_BGR2GRAY)
    
    #img_gray = cv2.cvtColor(pom, cv2.COLOR_RGB2GRAY)
    #pom1=pom>120
    
    pom1=cv2.erode(pom, kernel, iterations=1)
    #display_image(pom1)
    #plt.figure()
    lines = cv2.HoughLinesP(pom1,1,np.pi/180,60,50,50)
    
    x1,y1,x2,y2=get_koord(lines)
    plus=Tacka(x1,y1,x2,y2)
    
    print(plus.x1,plus.y1,plus.x2,plus.y2)
    pom2=frame[:,:,1]
    pom3=cv2.erode(pom2, kernel, iterations=1)
    #display_image(pom3)
    #plt.figure()
    lines = cv2.HoughLinesP(pom3,1,np.pi/180,60,50,50)
    
    x1,y1,x2,y2=get_koord(lines)
    minus=Tacka(x1,y1,x2,y2)
    print(minus.x1,minus.y1,minus.x2,minus.y2)
    binarna=image_bin(image_gray(frame))
    #print(binarna)
    #binarna2=cv2.erode(binarna, cv2.getStructuringElement(cv2.MORPH_CROSS, (2,2)), iterations=1)
    binarna3=cv2.erode(binarna, cv2.getStructuringElement(cv2.MORPH_CROSS, (2,2)), iterations=1)
    binarna3=cv2.dilate(binarna3, kerneldil, iterations=1)
    
    #print(frame)
    k_size = 2
    k = (1./k_size*k_size) * np.ones((k_size, k_size))
    image_blur = signal.convolve2d(binarna, k)
    #plt.imshow(image_blur, 'gray')
    #display_image(binarna)
    #plt.figure()
    
    binar=image_bin(image_blur)
    #print(binar)
    #display_image(binar)
    #plt.figure()
    aaaaaaaaaaaaaa=binarna
    #display_image(binarna3)
    #plt.figure()
    #image_orig, sorted_regions=select_roi(frame,binarna,plus)
    
    #display_image(image_orig)
    #plt.figure()
    
    #x,y=hist(sorted_regions[0])
    #print(y[0])
    
    #for reg in sorted_regions:
        
        #display_image(reg)
        #plt.figure()
    
    #print(lines[0][0][2])
    #for line in lines:
        #print(line)
        #print(line[0][0])
        #print(line[1])
    
    
    #print(lines)
    #for x1,y1,x2,y2 in lines[0]:
        #print(x1,y1,x2,y2)
        #cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),2)
    broj_piksela=[0,0]
    broj_piksela_minus=[0,0]
    konacan_rez=0
    regioni_plus=[]
    regioni_minus=[]
    
    broj=0
    
    while True:
        
        frame_num += 1
        ret_val, frame = cap.read()
        if not ret_val:
            break
        binarna=image_bin(image_gray(frame))
        k_size = 2
        k = (1./k_size*k_size) * np.ones((k_size, k_size))
        image_blur = signal.convolve2d(binarna, k)
        #img_gray = cv2.cvtColor(image_blur, cv2.COLOR_RGB2GRAY)
        #pomocna=
        #height, width = image_blur.shape[0:2]
        #image_binary1 = np.ndarray((height, width), dtype=np.uint8)
        #ret,image_binary1 = cv2.threshold(image_blur, 160, 255, cv2.THRESH_BINARY)
        #image_binary1=image_blur
        
        #image_blur=img_as_ubyte(image_blur)
        image_blur=np.uint8(image_blur)
        binarna=image_bin(image_blur)
        #binarna=cv2.erode(binarna, cv2.getStructuringElement(cv2.MORPH_CROSS, (2,2)), iterations=1)
        bbbbbbbbbbbbbbb=binarna
        #binarna=cv2.dilate(binarna, kerneldil, iterations=1)
        #binarna=cv2.erode(binarna, cv2.getStructuringElement(cv2.MORPH_CROSS, (2,2)), iterations=1)
       
        #binarna=cv2.erode(binarna, kerneldil, iterations=1)
        #binarna=cv2.dilate(cv2.erode(binarna, kernel, iterations=1), kerneldil, iterations=1)
        #binarna = cv2.morphologyEx(binarna, cv2.MORPH_OPEN, kernel)
    
        image_orig, sorted_regions=select_roi(frame,binarna,plus)
        #cv2.imshow("sasadasds",image_orig)

        
        for region in sorted_regions:        
             x,y=hist(region)
             if jedinstven(y,broj_piksela):
                 #print(y[-1])
                 broj+=1
                 broj_piksela.append(y[-1])
                 #display_image(image_orig)
                 #plt.figure()
    #             display_image(binarna)
    #             plt.figure()
                 regioni_plus.append(region)
                 
        image_orig_minus, sorted_regions_minus=select_roi(frame,binarna,minus)
        
        for region in sorted_regions_minus:        
             x,y=hist(region)
             if jedinstven(y,broj_piksela_minus):
                 #print(y[-1])
                 broj+=1
                 broj_piksela_minus.append(y[-1])
                 #display_image(image_orig_minus)
                 #plt.figure()
                 regioni_minus.append(region)
    
             
        
        #if frame_num==500:
        #    break
    
        #if not frame_num %3==0:
         #   continue
            
        #display_image(frame)
        #plt.figure()
        
    
    #print(frame_num)  
    #print(broj)  
    cap.release()
    
    alphabet = [0,1,2,3,4,5,6,7,8,9]
    rezultat_plus = ann.predict(np.array(prepare_for_ann(regioni_plus),np.float32))
    niz_plus=display_result(rezultat_plus,alphabet)
    rezultat_minus = ann.predict(np.array(prepare_for_ann(regioni_minus),np.float32))
    niz_minus=display_result(rezultat_minus,alphabet)
    
    for broj in niz_plus:
        konacan_rez+=broj
        
    for broj in niz_minus:
        konacan_rez-=broj
        
    #print(display_result(rezultat_minus,alphabet))
    print(konacan_rez)
    file.write('video-'+str(i)+'.avi\t' + str(konacan_rez)+'\r')

file.close()

