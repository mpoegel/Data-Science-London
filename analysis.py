# ------------------------------------------------------------------------------
# Kaggle
# Data Science London
#
# ------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

def PCA(A):
    ''' Returns the principal components of A
    '''
    A_mean = np.mean(A, axis=0)
    A = A - A_mean
    covariance_matrix = np.cov(A.T)
    eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
    new_index = np.argsort(eigen_values)[::-1]
    eigen_vectors = eigen_vectors[:,new_index]
    eigen_values = eigen_values[new_index]
    principal_components = np.dot(eigen_vectors.T, A.T).T

    return eigen_vectors, eigen_values, principal_components


def meanClassifier(Cp, Cm):
    ''' Calculate the mean classifier.
        Returns w,t: w (normal) = the difference of the means
                     t (threshold) = the midpoint of the means projected onto w
    '''
    meanp = Cp.mean(axis=0)
    meanm = Cm.mean(axis=0)

    w = (meanp - meanm)
    w = w / np.linalg.norm(w)

    t = (np.dot(meanp,w) + np.dot(meanm,w)) / 2

    return w, t


def fisherClassifier(Cp, Cm):
    ''' Calculate the fisher classifier.
        Returns w,t: w = normal
                     t = threshold
    '''
    meanp = Cp.mean(axis=0)
    meanm = Cm.mean(axis=0)

    psize = Cp.shape
    msize = Cm.shape
    Bp = Cp - np.ones((psize[0],1)) * meanp
    Bm = Cm - np.ones((msize[0],1)) * meanm
    Sw = np.dot(Bp.T,Bp) + np.dot(Bm.T,Bm)

    w = np.linalg.solve(Sw, (meanp - meanm).T)
    w = w / np.linalg.norm(w)

    t = (np.dot(meanp,w) + np.dot(meanm,w)) / 2

    return (w,t)


def makePredictions(data, w, t, file_name=''):
    ''' Make predictions for data and write predictions to file
        Returns the predictions.
    '''
    # projections and classifications of the data
    projections = np.dot(data,w)
    predictions = np.array([ int(x < t) for x in projections ])
    # if a file_name is given, then write to that file
    if (file_name != ''):
        f = open(file_name, 'w')
        # print a header to the file
        f.write("Id,Solution\n")
        for i in range(len(predictions)):
            f.write("{0},{1}\n".format(i+1,predictions[i]))

    return predictions


def classHist(Cp, Cm, t, title, err):
    ''' Return a Histogram of the projections.
    '''
    fig, ax = plt.subplots()
    colors = ['red', 'blue']
    labels = ['Class 1', 'Class 0']
    data = [Cp, Cm]
    n_bins = 20
    err_str = ", Error: " + str(round(err*100,2)) + '%'
    # plot the histogram
    ax.hist(data, n_bins, histtype='bar', color=colors, label=labels)
    ax.legend()
    ax.set_title(title + err_str)
    # add a vertical line at the threshold
    plt.axvline(x=t, linestyle='--', color='black')
    return fig


# ------------------------------------------------------------------------------
def main():
    ''' Create models using the Mean and Fisher LDA Methods to predict the
            classes of the test data.
    '''
    # load the data from the file
    train_file_name = "data/train.csv"
    test_file_name = "data/test.csv"
    trainLabels_file_name = "data/trainLabels.csv"

    trainData = np.loadtxt(train_file_name, delimiter=",")
    testData = np.loadtxt(test_file_name, delimiter=",")
    trainLabels = np.loadtxt(trainLabels_file_name, delimiter=",")

    # divide the training data by class
    class1_train = []
    class0_train = []
    for x in range(len(trainData)):
        if (trainLabels[x] == 1):
            class1_train.append(trainData[x])
        else: # trainLabels[x] == 0
            class0_train.append(trainData[x])
    class1_train = np.array(class1_train)
    class0_train = np.array(class0_train)

    # calculate the mean classifier
    (w, t) = meanClassifier(class1_train, class0_train)
    # scalar projections of the training data onto w
    c1_proj = np.dot(class1_train, w)
    c0_proj = np.dot(class0_train, w)
    # calculate the training error
    err = ([ int(x <= t) for x in c1_proj ],
           [ int(x >= t) for x in c0_proj ])
    err = (sum(err[0])/len(err[0]) + sum(err[1])/len(err[1])) / 2
    fig1 = classHist(c1_proj, c0_proj, t, "Mean Method", err)
    # make predictions based off the mean model
    mean_method_file_name = "MeanMethod_Predictions.txt"
    makePredictions(testData, w, t, mean_method_file_name)

    # calculate the fisher classifier
    (w, t) = fisherClassifier(class1_train, class0_train)
    # calculate the scalar projections on the training data
    c1_proj = np.dot(class1_train, w)
    c0_proj = np.dot(class0_train, w)
    # calculate the training error
    err = ([ int(x <= t) for x in c1_proj ],
           [ int(x >= t) for x in c0_proj ])
    err = (sum(err[0])/len(err[0]) + sum(err[1])/len(err[1])) / 2
    fig2 = classHist(c1_proj, c0_proj, t, "Fisher Method", err)
    # make predictions based off of the Fisher model
    fisher_file_name = "Fisher_Predictions.txt"
    makePredictions(testData, w, t, fisher_file_name)

    # Visualize the data by plotting the first two principal components
    eigen_vectors, eigen_values, principal_components = PCA(trainData)
    plt.figure(3)
    # break up the principal components by class
    c1_pc = np.array([ trainData[x] for x in range(len(trainData)) \
                            if trainLabels[x] == 1 ])
    c0_pc = np.array([ trainData[x] for x in range(len(trainData)) \
                            if trainLabels[x] == 0 ])
    # plot class 1 in red and class 0 in blue
    plt.plot(c1_pc[:,0], c1_pc[:,1], 'ro')
    plt.plot(c0_pc[:,0], c0_pc[:,1], 'bo')
    plt.title("Training Data - First 2 Principal Components")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    # show the explained variance by each principal component
    explained_variance = np.cumsum(eigen_values) / sum(eigen_values) * 100
    plt.figure(4)
    plt.bar(np.arange(len(explained_variance)), explained_variance)
    plt.title("Explained Variance of Principal Components")
    plt.xlabel("Principal Components")
    plt.ylabel("Percent Explained Explained")


    plt.show()

    return 0


# ------------------------------------------------------------------------------
if (__name__ == '__main__'):
    main()
