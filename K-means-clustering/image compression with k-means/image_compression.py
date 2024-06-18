import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(r'C:\Users\Ayaan\Desktop\Machine Learning\K-means clustering')
from kmeans import run_kmeans
from PIL import Image

def main(img_name,k,num_init=1,num_iters=10):
    img = np.asarray(Image.open(img_name))
    X = np.reshape(img, (img.shape[0]*img.shape[1],3))
    J, mu, c = run_kmeans(X,k,num_init,num_iters)
    mu = mu.astype(np.uint8)
    new = mu[c]
    new = np.reshape(new, img.shape)
    plt.imsave(img_name[:-4]+'_new'+img_name[-4:], new)
    plt.close()

main('bird_small.png', 16)
#by setting k=2 it acts as a binarizer
main('mona_lisa.png', 2)
main('ayaan.jpg', 2)
