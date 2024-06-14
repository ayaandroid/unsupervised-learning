import numpy as np
import matplotlib.pyplot as plt
import random

def run_kmeans(X,k,num_init,num_iters):
    #X: np.ndarray, X.shape = (m,n) -> m training examples, n features
    #k: number of clusters
    #num_init: number of random initializations
    #num_iters: number of iterations
    #returns min_J: min value of cost J over all initializations,
    #best_mu: np.ndarray, best_mu.shape = (k,n): positions of cluster centroids in clustering with lowest J
    #best_c: np.ndarray, c.shape = (m,): c[i] = index(0...k-1) of centroid closest to X[i] in clustering with lowest J

    m,n = X.shape
    
    min_J = np.inf
    best_mu = np.zeros((k,n))
    best_c = np.zeros(m)
    
    for _ in range(num_init):
        
        #randomly initialize mu
        mu = X[random.sample(range(m),k)]
        
        #initialize c (setting nearest centroid of all points as the first centroid)
        c = np.zeros(m, dtype=int)
        
        for __ in range(num_iters):
            
            #assign points to nearest centroid
            for i in range(m):
                min_dist = np.linalg.norm(X[i]-mu[c[i]])
                for j in range(k):
                    dist = np.linalg.norm(X[i]-mu[j])
                    if dist < min_dist:
                        c[i] = j
                        min_dist = dist

            #shift each centroid to mean of points assigned to it     
            cnt = np.zeros(k)
            mu = np.zeros((k,n))
            for i in range(m):
                mu[c[i]] += X[i]
                cnt[c[i]] += 1
            #cnt[j] = number of points assigned to jth centroid
            #mu[j] = sum of points assigned to jth centroid  
            for j in range(k):
                if cnt[j] != 0:
                    #shift to mean
                    mu[j] = mu[j]/cnt[j]
                else:
                    #if no points have jth centroid as their closest, randomly initialize it
                    mu[j] = random.choice(X)

        #compute J
        J = 0
        for i in range(m):
            J += (np.linalg.norm(X[i]-mu[c[i]]))**2
        J /= m
        
        if J < min_J:
            best_mu = mu.copy()
            best_c = c.copy()
            min_J = J
            
    return min_J, best_mu, best_c

#Example
X_train = np.array([[10,1],[8,9],[1,1],[1,2],[7,6],[8,2],[9,3],[6,8],[3,2],
                    [3,3],[7,8],[10,4],[11,2]])
#n=2 features (so that we can plot it on xy plane)

#run k means with k=3 clusters
k = 3
J, mu, c = run_kmeans(X_train,k,10,10)

#plot
colors = ['r','g','b']
for j in range(k):
    cluster_j = X_train[c==j]
    plt.scatter(cluster_j[:,0],cluster_j[:,1],color = colors[j])
plt.scatter(mu[:,0],mu[:,1],color = 'k',marker='x')
plt.savefig('example.png')
plt.close()
