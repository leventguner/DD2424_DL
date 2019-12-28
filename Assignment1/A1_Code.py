import numpy as np
import matplotlib.pyplot as plt
import pickle
plt.style.use('ggplot')



parameters = {'mu': 0, 'sigma': 0.01, 'h': 1e-6, 'n_batch': 100, 'n_epochs': 40, 'lamda': 0, 'lr': 0.01}


mu = parameters['mu']
sigma = parameters['sigma']
h = parameters['h']
n_batch = parameters['n_batch']
n_epochs = parameters['n_epochs']
lamda = parameters['lamda']
eta = parameters['lr']

def load_data(file):
    """
    Input: file / output: X and y arrays, also one-hot encoding
    """
    path = '../cifar_10/{}'.format(file)
    with open(path, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    f.close()
    X = np.array(data[b'data'] / 255).T
    y = np.array(data[b'labels'])
    N = X.shape[1]
    k = int(max(y)+1)
    Y = np.zeros((k, N))
    Y[y, np.arange(N)] = 1
    return X, Y, y

X, Y, y = load_data("data_batch_1")

XVal, YVal, yVal = load_data("data_batch_2")
XTest, YTest, yTest = load_data("test_batch")

N = X.shape[1]
k = int(max(y)+1)
d = X.shape[0]

def initials():
    """
    Define initial W and b
    """
    W = np.random.normal(mu, sigma, (k, d))
    b = np.random.normal(mu, sigma, (k, 1))
    return W,b

W,b = initials()

def softmax(S):
    """
    Input:Wx+b / Output:Softmax
    """
    soft = np.exp(S) / np.sum(np.exp(S), axis=0)
    return soft


def evaluate_classifier(X, W=W, b=b):
    """
    Evaluate probabilities with softmax
    """
    probas = softmax(np.dot(W, X) + b)
    return probas


def compute_cost(X, Y, W, b):
    """
    Compute the cost
    """
    pred = evaluate_classifier(X, W, b)
    num_e = X.shape[1]
    L2reg = lamda * np.power(W, 2).sum()
    y_p = np.multiply(Y, pred).sum(axis=0)

    cross_ent = np.sum(-np.log(y_p))
    #
    cost = cross_ent / num_e + L2reg
    return cost


def compute_accuracy(predictions, y):
    """
    Compute the accuracy
    """

    corr = len(np.where(predictions == y)[0])
    accuracy = (corr / len(y)) * 100
    return accuracy


def compute_grad_analytic(X, Y, W, b):
    """
    Compute the gradient analytically, based on the centered difference formula
    """
    preds = evaluate_classifier(X, W, b)

    ones = np.ones((N, 1))
    grad = - (Y - preds)
    grad_b = np.dot(grad, ones/N)

    grad_W = np.dot(grad, X.T)/N
    grad_W = np.add(grad_W, (2 * lamda * W))

    return grad_W, grad_b


def compute_grad_numeric(X, Y, W, b):
    """
    Compute the gradient numerically
    """
    grad_W = np.zeros((k, d))
    grad_b = np.zeros((k, 1))

    for i in range(b.shape[0]):
        b[i] -= h
        cost1 = compute_cost(X, Y, W, b)
        b[i] += 2*h
        cost2 = compute_cost(X, Y, W, b)
        grad_b[i] = (cost2 - cost1) / (2*h)
        b[i] -= h

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W[i, j] -= h
            cost1 = compute_cost(X, Y, W, b)
            W[i, j] += 2*h
            cost2 = compute_cost(X, Y, W, b)
            grad_W[i, j] = (cost2 - cost1) / (2*h)
            W[i, j] -= h

    return grad_W, grad_b


def check_gradients(W, b):
    """
    Check if analytic and numeric gradients are the same
    """

    grad_w_a, grad_b_a = compute_grad_analytic(X[:20,0:1], Y[:,0:1], W[:,:20], b)

    grad_w_n, grad_b_n = compute_grad_numeric(X[:20,0:1], Y[:,0:1], W[:,:20], b)
    print("grad(W)")
    print("-------")
    print('Sum of absolute diff.: ', sum(np.abs(grad_w_a - grad_w_n)))

    print("\ngrad(B)")
    print("-------")
    print('Sum of absolute diff.: ', sum(np.abs(grad_b_a - grad_b_n)))


def fit(X,Y,y,XVal, YVal, yVal,W=W,b=b):
    """
    Fit the model and make predictions
    """
    accuracy_values = []
    accuracy_val_values = []
    cost_values = []
    cost_val_values = []

    for epoch in range(n_epochs):
        for j in range(int(N / n_batch)):

            s_st,j_end = j * n_batch,n_batch*(j+1)

            X_batch = X[:, s_st:j_end]
            Y_batch = Y[:, s_st:j_end]
            w_upd, b_upd = compute_grad_analytic(X_batch, Y_batch, W, b)

            W = W - eta * w_upd.reshape((k, d))
            b = b - eta * b_upd.reshape((k, 1))


        cost = compute_cost(X, Y, W, b)
        probas = evaluate_classifier(X, W, b)
        predictions = np.argmax(probas, axis=0)
        accuracy = compute_accuracy(predictions, y)
        cost_values.append(cost)
        accuracy_values.append(accuracy)
        print("Epoch: {}".format(epoch))
        print("Training cost = {}".format(cost))
        print("Training accuracy = {}".format(accuracy))
        print("-----------------")

        probas_val = evaluate_classifier(XVal, W, b)
        cost_val = compute_cost(XVal, YVal, W, b)
        predictions_val = np.argmax(probas_val, axis=0)
        accuracy_val = compute_accuracy(predictions_val, yVal)
        cost_val_values.append(cost_val)
        accuracy_val_values.append(accuracy_val)
        print("Validation cost = {}".format(cost_val))
        print("Validation Accuracy = {}\n".format(accuracy_val))

    return cost_values, cost_val_values, accuracy_values, accuracy_val_values, W,b


cost, costval, acc, accval, W_l,b_l = fit(X,Y,y,XVal, YVal, yVal)


def predict(X):
    probas = softmax(np.dot(W, X) + b)
    preds = np.argmax(probas, axis=0)
    return probas, preds

probas_test,predictions_test = predict(XTest)
test_accuracy = compute_accuracy(predictions_test, yTest)
print("Test accuracy: ", test_accuracy)


#draw weight images
for i in range(k):
    w_im = W_l[i,:].reshape((32, 32, 3), order='F')
    w_im = ((w_im - w_im.min()) / (w_im.max() - w_im.min()))
    w_im = np.rot90(w_im, -1)
    plt.figure()
    plt.imshow(w_im)
    plt.grid(False)
    plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off',labelbottom=False,labelleft=False)
    plt.title("Class " + str(i))
    plt.savefig('lamda_{}_eta_{}_class_{}.png'.format(lamda,eta,i))


plt.figure()
plt.plot(np.arange(len(cost)), cost)
plt.plot(np.arange(len(costval)), costval)
plt.title('Lambda = {}, lr = {}'.format(lamda,eta))
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Training','Validation'],loc='best')
plt.savefig('cost_lamda_{}_eta_{}.png'.format(lamda,eta))

plt.figure()
plt.plot(np.arange(len(acc)), acc)
plt.plot(np.arange(len(accval)), accval)
plt.title('Lambda = {}, lr = {}'.format(lamda,eta))
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Training','Validation'],loc='best')
plt.savefig('accuracy_lamda_{}_eta_{}_{}.png'.format(lamda,eta,np.round(test_accuracy,2)))

probas_test,predictions_test = predict(XTest)


def show_w(W, k):
    for i in range(k):
        w_image = W[i, :].reshape((32, 32, 3), order='F')
        w_image = ((w_image - w_image.min()) / (w_image.max() - w_image.min()))
        w_image = np.rot90(w_image, 3)
        plt.imshow(w_image)
        plt.tick_params(axis='both',  which='both',bottom='off',top='off',left='off',right='off')
        plt.title("Class " + str(i))
        plt.show()

plt.figure()
show_w(W_l, 10)

plt.show()