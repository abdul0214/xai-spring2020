import numpy as np
import sklearn
import sklearn.cluster

# def calc_identity(exp1, exp2):
#     dis = np.linalg.norm(exp1-exp2, axis = 1)
#     total = dis.shape[0]
#     true = np.where(abs(dis)<1e-8)[0].shape[0]
#     score = (total-true)/total
#     return score*100

def calc_similarity(coefs1,X_test):
    from sklearn.preprocessing import normalize
    from sklearn.cluster import DBSCAN
    from sklearn.metrics.pairwise import euclidean_distances
    from sklearn import preprocessing 
    import numpy as np
    X_test = preprocessing.normalize(X_test, norm='l2')
    coefs1 = preprocessing.normalize(coefs1, norm='l2')
    clustering = DBSCAN(eps=0.02,min_samples=2).fit(X_test)
    cluster_labels = clustering.labels_
    unique_labels=np.unique(cluster_labels)
    mean_distances=[]
    for cluster in unique_labels:
        indexes_of_cluster=np.where(cluster_labels == cluster)[0]
        exps_of_cluster = [coefs1[i] for i in indexes_of_cluster] 
        mean_distances.append(np.average(euclidean_distances(exps_of_cluster)))
        
    return np.average(mean_distances)

def calc_correctness(exps, labels, true_labels) :
    incorrect = 0
    import numpy as np
    exps=np.asarray(exps)
    labels=np.asarray(labels)
    total = labels.shape[0]
    label_values = np.unique(labels)
    unique_exps = np.unique(exps, axis=0)
    correct_exps = exps[(np.where(y1==y_test)[0])]
    unique_exps = np.unique(correct_exps, axis=0)
    similar_exps = [] 
    for exp in unique_exps:
        indexes_of_exp = np.where((exps == exp).all(axis=1))[0]
        for j in indexes_of_exp:
            if (labels[j] != labels[indexes_of_exp[0]]):
                incorrect += 1
    return (len(unique_exps)-incorrect)/(len(unique_exps))


def calc_identity(exp1, exp2):
    # dis = np.array(exp1)==np.array(exp2)
    # dis = dis.all(axis = 1)
    dis = np.array([np.array_equal(exp1[i],exp2[i]) for i in range(len(exp1))])
    total = dis.shape[0]
    true = np.sum(dis)
    score = (total-true)/total
    return score*100, true, total


def calc_separability(x_test, exp):
    import pandas as pd
    import numpy as np
    x_test = pd.DataFrame(x_test)
    exp = np.asarray(exp)
    dissimilar_instances=len(x_test.drop_duplicates())
    similar_exps = 0
    for i in range(exp.shape[0]):
        for j in range(exp.shape[0]):
            if i == j:
                continue
            eq = np.array_equal(exp[i],exp[j])
            if eq:
                similar_exps += 1
    total = exp.shape[0]
    score = (dissimilar_instances-similar_exps)/dissimilar_instances
    return dissimilar_instances,similar_exps,score*100


def calc_compactness(exps):
    import pandas as pd
    import numpy as np
    exps = np.asarray(exps)
    scores=[]
    for exp in exps:
        scores.append((len(exp)-np.count_nonzero(exp))/len(exp))
    return (np.average(scores))


def calc_stability(exp, labels):
    import numpy as np
    exp=np.asarray(exp)
    labels=np.asarray(labels)
    total = labels.shape[0]
    label_values = np.unique(labels)
    n_clusters = label_values.shape[0]
    init = np.array([[np.average(exp[np.where(labels == i)], axis = 0)] for i in label_values]).squeeze()
    ct = sklearn.cluster.KMeans(n_clusters = n_clusters, n_jobs=5, random_state=1, n_init=10, init = init)
    ct.fit(exp)
    error = np.sum(np.abs(labels-ct.labels_))
    if error/total > 0.5:
        error = total-error
    return error, total

def calc_stability(exp, labels):
    import numpy as np
    exp=np.asarray(exp)
    labels=np.asarray(labels)
    total = labels.shape[0]
    label_values = np.unique(labels)
    n_clusters = label_values.shape[0]
    init = np.array([[np.average(exp[np.where(labels == i)], axis = 0)] for i in label_values]).squeeze()
    ct = sklearn.cluster.KMeans(n_clusters = n_clusters, n_jobs=5, random_state=1, n_init=10, init = init)
    ct.fit(exp)
    error = np.sum(np.abs(labels-ct.labels_))
    if error/total > 0.5:
        error = total-error
    return error, total

def calc_correctness(exp, labels):
    import numpy as np
    exp=np.asarray(exp)
    labels=np.asarray(labels)
    total = labels.shape[0]
    label_values = np.unique(labels)
    uniqu
    n_clusters = label_values.shape[0]
    init = np.array([[np.average(exp[np.where(labels == i)], axis = 0)] for i in label_values]).squeeze()
    ct = sklearn.cluster.KMeans(n_clusters = n_clusters, n_jobs=5, random_state=1, n_init=10, init = init)
    ct.fit(exp)
    error = np.sum(np.abs(labels-ct.labels_))
    if error/total > 0.5:
        error = total-error
    return error, total



# def calc_separability(x_test, exp):
#     #x_test = np.unique(x_test)
#     dissimilar_instances=len(np.unique(x_test))
#     print("dissimilar_instances ",dissimilar_instances)
#     similar_exps = 0
#     for i in range(exp.shape[0]):
#         for j in range(exp.shape[0]):
#             if i == j:
#                 continue
#             eq = np.array_equal(exp[i],exp[j])
#             if eq:
#                 similar_exps += 1
#     total = exp.shape[0]
#     #score = 100*abs(wrong)/total**2
#     return dissimilar_instances,similar_exps,total**2
