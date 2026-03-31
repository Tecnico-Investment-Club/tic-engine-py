from scipy.special import comb

def accuracy_bagging(N = 100, p = 1/3, k=3):
    """
    N: number of independent classifiers
    p: probability of each classifier making a right prediction 
    K: number of classes
    Returns the probability of the bagging classifier making a right prediction.
    """
    p_=0
    for i in range(0,int(N/k)+1):
        p_+=comb(N,i)*p**i*(1-p)**(N-i)
    return 1-p_