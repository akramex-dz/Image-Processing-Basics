import tools
import matplotlib.pyplot as plt
import numpy as np
import cv2

def get_gaussian_filtre(dimension=3,sigma=0.5):
    """return the result after apply a gaussian filtre with a given dimension """
    kernel = tools.gaussian_mask(size=dimension,sigma=sigma)
    return kernel

def get_mean_filter(dimension=3):
    """return the result after apply a mean filtre with a given dimension """
    kernel= np.ones((dimension, dimension), np.float32) / dimension**2
    return kernel

def plot_img(img,title):
    fig,ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    ax.set_title(title)

def filter_analysis(img, kernel, cmap = None):
    """apply filtre on imageand compare between original and filtred image"""
    filtred_im =apply_filter_to_single_channel(img,kernel)
    fig,ax = plt.subplots(1,2)
#     ploting original img
    ax[0].imshow(img, cmap = cmap)
    ax[0].set_title("original")
#     ploting filtred img
    ax[1].imshow(filtred_im, cmap = cmap)
    ax[1].set_title('filtred image')

def apply_filter_to_single_channel(img,kernel):
    dimK = kernel.shape
    return tools.Conv2D(tools.add_padding(img,((dimK[0]-1)//2,(dimK[1]-1)//2)),kernel)

def apply_filter_to_colored_img(img,kernel):
#     dstack build ndarray on the third axis
#    return np.dstack([apply_filter_to_single_channel(img[:,:,z],kernel) for z in range(3)])
    return cv2.filter2D(src=img, ddepth=-1, kernel=kernel) 

def filter_analysis_colored(img,kernel):
    """apply filtre on imageand compare between original and filtred image"""
   
    filtred_im = apply_filter_to_colored_img(img,kernel)
    fig,ax = plt.subplots(1,2)
#     ploting original img
    ax[0].imshow(img)
    ax[0].set_title("original")
#     ploting filtred img
    ax[1].imshow(filtred_im)
    ax[1].set_title('filtred image')

def get_Gx(img):
#     get img conv Sx filter
    Sx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    return apply_filter_to_single_channel(img,Sx)

def get_Gy(img):
#     get img conv Sy filter
    Sy = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    return apply_filter_to_single_channel(img,Sy)

# filtre laplace
def get_L(img):
    L = np.array([[0,1,0],[1,-4,1],[0,1,0]])
    return apply_filter_to_single_channel(img,L)

def module_grad(img):
    Gx = get_Gx(img)
    Gy = get_Gy(img)
    result=np.sqrt(Gx**2+Gy**2)
    return result

def direct_grad(img):
    Gx = get_Gx(img)
    Gy = get_Gy(img)
    result=np.arctan(Gy/Gx)
    return result

def sobel(img):
    Gx = get_Gx(img)
    Gy = get_Gy(img)
    grad_direct = np.arctan(Gy/Gx)
    grad_module = np.sqrt(Gx**2+Gy**2)
    return grad_module, grad_direct

# Fonction pour détecter les contours
def detecter_contours(Lr, seuil):
    contours = np.zeros_like(Lr)
    max_Lr = Lr.max()
    min_Lr = Lr.min()

    for i in range(1,Lr.shape[0]-1):
        for j in range(1,Lr.shape[1]-1):
            fenetre_centree = Lr[i-1:i+2,j-1:j+2]
            max_fenetre_centree = fenetre_centree.max()
            min_fenetre_centree = fenetre_centree.min()

            if max_fenetre_centree > 0 and min_fenetre_centree < 0 and (max_fenetre_centree - min_fenetre_centree) > seuil:
                contours[i,j] = 255

    return contours

    
def non_maximum_suppression(gradient_module, gradient_orientation):
    # Conversion des angles en degrés
    gradient_orientation = np.degrees(gradient_orientation)
    
    region_1 = (0, 45)
    region_2 = (45, 90)
    region_3 = (90, 135)
    region_4 = (135, 180)
    
    
    # Copie de l'image des modules de gradient
    inms = np.copy(gradient_module)
    
    # Parcours de chaque pixel de l'image
    for i in range(1, gradient_module.shape[0] - 1):
        for j in range(1, gradient_module.shape[1] - 1):
            # Récupération de l'orientation du gradient du pixel courant
            orientation = gradient_orientation[i, j]
            
            # Initialisation des coordonnées des pixels voisins dans la direction de l'orientation
            x1, y1 = 0, 0
            x2, y2 = 0, 0
            
            # Calcul des coordonnées des pixels voisins dans la direction de l'orientation
            if region_1[0] <= orientation < region_1[1] or region_4[0] <= orientation < region_4[1]:
                x1, y1 = i, j + 1
                x2, y2 = i, j - 1
            elif region_2[0] <= orientation < region_2[1]:
                x1, y1 = i - 1, j + 1
                x2, y2 = i + 1, j - 1
            elif region_3[0] <= orientation < region_3[1]:
                x1, y1 = i - 1, j
                x2, y2 = i + 1, j
            
            # Comparaison des modules de gradient du pixel courant avec ses voisins dans la direction de l'orientation
            if gradient_module[i, j] < gradient_module[x1, y1] or gradient_module[i, j] < gradient_module[x2, y2]:
                inms[i, j] = 0
    
    return inms

