# ------------------------------------------------------------------------------
# Kaggle
# Data Science London
#
# ------------------------------------------------------------------------------
import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt

def meanClassifier(Cp, Cm):
    ''' Calculate the mean classifier.
        Returns w,t: w (normal) = the difference of the means
                     t (threshold) = the midpoint of the means projected onto w
    '''
    meanp = Cp.mean(axis=0)
    meanm = Cm.mean(axis=0)

    w = (meanp - meanm)
    w = w / linalg.norm(w)

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

    w = linalg.solve(Sw, (meanp - meanm).T)
    w = w / linalg.norm(w)

    t = (np.dot(meanp,w) + np.dot(meanm,w)) / 2

    return (w,t)


def classifyData(data, w, t):
    pass

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

    plt.show()

    return 0


# ------------------------------------------------------------------------------
if (__name__ == '__main__'):
    main()
