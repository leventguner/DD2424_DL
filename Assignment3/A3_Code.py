import numpy as np
import matplotlib.pyplot as plt
import pickle

plt.style.use('ggplot')


def load_data(file):
    """
    Input: file / output: X and y arrays, also one-hot encoding
    """
    path = '../cifar_10/{}'.format(file)
    with open(path, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    f.close()
    X = np.array(data[b'data'] / 255).T

    mean_s = np.mean(X, axis=0)
    std_d = np.std(X, axis=0)
    X = X - mean_s
    X = X / std_d

    y = np.array(data[b'labels'])
    N = X.shape[1]
    k = int(max(y) + 1)
    Y = np.zeros((k, N))
    Y[y, np.arange(N)] = 1
    return X, Y, y


Xa,Ya,ya = load_data("data_batch_1")


for i in [2,3,4,5]:
    Xb, Yb, yb = load_data("data_batch_{}".format(i))
    Xa = np.concatenate([Xa, Xb],axis=1)
    Ya = np.concatenate([Ya, Yb],axis=1)
    ya = np.concatenate([ya, yb])

val_size=5000
X,Y,y = Xa[:,:-val_size],Ya[:,:-val_size],ya[:-val_size]
X_val, Y_val, y_val = Xa[:,-val_size:],Ya[:,-val_size:],ya[-val_size:]
print(X.shape,Y.shape,y.shape)
print(X_val.shape,Y_val.shape,y_val.shape)

X_test, YTest, y_test = load_data("test_batch")

N = X.shape[1]
k = int(max(y)+1)
d = X.shape[0]


parameters = {
    'mu': 0,
    'h_size': 50,
    'h': 1e-5,
    'batch_size': 100,
    'lamda': 0.001,
    'lr_min': 1e-5,
    'lr_max': 1e-1,
    'cycles': 2,
    'n_layers':9,
    'alpha':0.7
}

mu = parameters['mu']
m = parameters['h_size']
h = parameters['h']
batch_size = parameters['batch_size']
lamda = parameters['lamda']

lr_min = parameters['lr_min']
lr_max = parameters['lr_max']
n_cycles = parameters['cycles']
n_layers = parameters['n_layers']
alpha = parameters['alpha']
mu_avg=None
var_avg=None
batch_norm=True

#hidden_nodes = [m]*(n_layers-1)
hidden_nodes = [50,30,20,20,10,10,10,10]
hidden_nodes = hidden_nodes+[k]

print('Nodes:',hidden_nodes)



def initials():
    sigma_weights_w = []
    b = [None] * n_layers
    gammas = [None] * (n_layers-1)
    betas = [None] * (n_layers-1)

    sigma_weights_w.append(1 / np.sqrt(d))

    for i in range(1, n_layers):
        sigma_weights_w.append(1 / np.sqrt(hidden_nodes[i - 1]))

    w = [np.zeros((2, 2))] * n_layers
    w[0] = np.random.normal(mu, sigma_weights_w[0], (hidden_nodes[0], d))
    b[0] = np.zeros((hidden_nodes[0], 1))

    for i in range(1, n_layers):
        w[i] = np.random.normal(mu, sigma_weights_w[i], (hidden_nodes[i], hidden_nodes[i - 1]))
        b[i] = np.zeros((hidden_nodes[i], 1))

    for i in range(n_layers-1):
        input_size = hidden_nodes[i]
        var = np.sqrt(2 / input_size)
        gammas[i] = np.random.normal(mu, var, (input_size, 1))
        betas[i] = np.random.normal(mu, sigma_weights_w[i], (hidden_nodes[i], 1))
    return w, b , gammas , betas

siggma=1e-1
def initials_sens():
    #sigma_weights_w = []
    b = [None] * n_layers
    gammas = [None] * (n_layers-1)
    betas = [None] * (n_layers-1)


    #sigma_weights_w.append(1 / np.sqrt(d))

    #for i in range(1, n_layers):
    #    sigma_weights_w.append(1 / np.sqrt(hidden_nodes[i - 1]))
    sigma_weights_w = [siggma]*n_layers
    w = [np.zeros((2, 2))] * n_layers
    w[0] = np.random.normal(mu, sigma_weights_w[0], (hidden_nodes[0], d))
    b[0] = np.zeros((hidden_nodes[0], 1))

    for i in range(1, n_layers):
        w[i] = np.random.normal(mu, sigma_weights_w[i], (hidden_nodes[i], hidden_nodes[i - 1]))
        b[i] = np.zeros((hidden_nodes[i], 1))

    for i in range(n_layers-1):
        input_size = hidden_nodes[i]
        var = np.sqrt(2 / input_size)
        gammas[i] = np.random.normal(mu, var, (input_size, 1))
        betas[i] = np.random.normal(mu, sigma_weights_w[i], (hidden_nodes[i], 1))
    return w, b , gammas , betas

def shuffle(x , y):
    indicies = np.arange(X.shape[1])
    np.random.shuffle(indicies)
    x_t = x.T
    y_t = y.T
    shuffled_x = x_t[indicies]
    shuffled_y = y_t[indicies]
    x = shuffled_x.T
    y = shuffled_y.T
    return x, y

def cyclic_lr(n_s, upd, lr_min=lr_min, lr_max=lr_max):
    lr_g = int(upd / (2 * n_s))
    diff = lr_max - lr_min

    if upd < 2 * lr_g * n_s + n_s:
        lr = lr_min + (upd - 2 * lr_g * n_s) / n_s * diff
    else:
        lr = lr_max - (upd - (2 * lr_g + 1) * n_s) / n_s * diff

    return lr





def batch_normalize(s, mean = None,  var = None ):

    epsilon = 0

    if mean is None:
        mean = np.mean(s, axis=1 , keepdims=True)
    if var is None:
        var = np.var(s, axis=1 , keepdims=False)

    d1 =  np.diag(np.sqrt(1/(var+epsilon)))
    d2 = s - mean
    s_h = np.dot(d1 , d2)
    return s_h

def update_mu_var(mu=None, var=None):
    global mu_avg
    global var_avg

    if mu_avg is None:
        mu_avg = mu

    else:
        mu_avg = [alpha * mu_avg[l] + (1-alpha) * mu[l] for l in range(len(mu))]

    if var_avg is None:
        var_avg = var
    else:
        var_avg = [alpha * var_avg[l] + (1-alpha) * var[l] for l in range(len(var))]

def softmax(S):
    """
    Input:Wx+b / Output:Softmax
    """
    soft = np.exp(S) / np.sum(np.exp(S), axis=0)
    return soft


def evaluate_classifier(X, Ws, bs, gammas,betas):
    activations = []
    scores = []
    means = []
    variances = []
    s_hs = []

    for l_no in range(n_layers-1):

        if l_no==0:
            score=np.dot(Ws[l_no], X) + bs[l_no]
            scores.append(score)
        else:
            score = np.dot(Ws[l_no], activations[l_no-1]) + bs[l_no]
            scores.append(score)

        if batch_norm:
            mu = np.mean(score, axis=1 , keepdims=True)
            means.append(mu)
            var = np.var(score, axis=1 , keepdims=False)
            variances.append(var)
            s_h = batch_normalize(score,mu,var)
            #means.append(mean)
            #variances.append(variance)
            s_hs.append(s_h)
            s_t = np.multiply(gammas[l_no], s_h) + betas[l_no]
            activations.append(np.maximum(0, s_t))

        else:
            activations.append(np.maximum(0, score))

    if batch_norm:
        update_mu_var(means, variances)

    #print(len(Ws),len(activations),len(bs),n_layers)
    S = np.dot(Ws[n_layers-1], activations[n_layers-2]) + bs[n_layers-1]
    scores.append(S)


    probas = softmax(S)
    preds = np.argmax(probas, axis=0)


    return activations, probas, preds , scores ,s_hs , means , variances


def compute_cost(X,Y,Ws,bs,gammas,betas,lamda):

    acts,probas,_,_,_,_,_ = evaluate_classifier(X,Ws,bs,gammas,betas)
    num_e = X.shape[1]

    wss = 0
    for i in range(n_layers):
        wss += np.sum(Ws[i] ** 2)

    L2 = lamda * wss
    y_p = np.multiply(Y, probas).sum(axis=0)


    cross_ent = np.sum(-np.log(y_p))

    loss = cross_ent / num_e
    cost = loss + L2

    return loss,cost

def compute_accuracy(predictions, y):
    """
    Compute the accuracy
    """

    corr = len(np.where(predictions == y)[0])
    accuracy = (corr / len(y))
    return accuracy


def batch_norm_backpass(g, scores, means):
    sigma1 = 1 / np.sqrt(np.mean(np.power((scores - means), 2), axis=1, keepdims=True))
    sigma2 = np.power(sigma1, 3)
    G1,G2 = g * sigma1, g * sigma2
    D = np.subtract(scores, means)
    c = np.sum(np.multiply(G2, D), axis=1, keepdims=True)
    p1 = np.subtract(G1, np.sum(G1, axis=1, keepdims=True) / scores.shape[1])
    p2 = np.multiply(D, c) / scores.shape[1]
    G = np.subtract(p1, p2)
    return G

def compute_grad_analytic(X, Y, Ws, bs, gammas, betas,lamda):
    ones = np.ones((X.shape[1], 1))

    grad_ws = [None]*n_layers
    grad_bs = [None]*n_layers
    grad_gammas = [None]*n_layers
    grad_betas = [None]*n_layers
    activations, probabilities, predictions, scores, sHats, means, variances = evaluate_classifier(X, Ws, bs, gammas,betas)

    g = - (Y - probabilities)
    #print(probabilities)
    grad_bs[-1] = np.dot(g, ones) / X.shape[1]
    #print(len(activations),layer)

    grad_ws[-1] = np.dot(g, activations[-1].T) / X.shape[1]
    grad_ws[-1] = np.add(grad_ws[-1], (2 * lamda * Ws[-1]))


    g = np.dot(Ws[-1].T, g)
    indicator = 1 * (activations[-1] > 0)
    g = np.multiply(g, indicator)

    np.dot(Ws[n_layers - 1], activations[n_layers - 2]) + bs[n_layers - 1]

    for l in reversed(range(n_layers-1)):
        if batch_norm:
            grad_gammas[l] = np.dot(np.multiply(g, sHats[l]), ones) / X.shape[1]
            grad_betas[l] = np.dot(g, ones) / X.shape[1]
            g = g * np.dot(gammas[l],ones.T)
            g = batch_norm_backpass(g,scores[l],means[l])

        if l == 0:
            grad_ws[l] = np.dot(g, X.T) / X.shape[1]
            grad_ws[l] = np.add(grad_ws[l], (2 * lamda * Ws[l]))
        else:
            grad_ws[l] = np.dot(g, activations[l - 1].T) / X.shape[1]
            grad_ws[l] = np.add(grad_ws[l], (2 * lamda * Ws[l]))

        grad_bs[l] = np.dot(g, ones) / X.shape[1]

        if l > 0:
            indicator = 1 * (activations[l-1] > 0)
            g = np.dot(Ws[l].T, g)
            g = np.multiply(g, indicator)

    return grad_bs, grad_ws, grad_gammas, grad_betas


def compute_grad_numeric(X, Y, Ws, bs, gammas, betas,lamda):
    grad_ws = [None] * (n_layers)
    grad_bs = [None] * (n_layers)
    grad_gammas = [None] * (n_layers)
    grad_betas = [None] * (n_layers)
    for i in range(n_layers-1):
        grad_bs[i] = np.zeros((hidden_nodes[i], 1))
        grad_gammas[i] = np.zeros((hidden_nodes[i], 1))
        grad_betas[i] = np.zeros((hidden_nodes[i], 1))

    grad_bs[n_layers-1] = np.zeros((k, 1))
    grad_gammas[n_layers - 1] = np.zeros((k, 1))
    grad_betas[n_layers - 1] = np.zeros((k, 1))

    grad_ws[0] = np.zeros((hidden_nodes[0], X.shape[0]))

    for i in range(1, n_layers-1):
        grad_ws[i] = np.zeros((hidden_nodes[i], hidden_nodes[i - 1]))

    grad_ws[n_layers-1] = np.zeros((k, hidden_nodes[n_layers - 2]))

    for l_no in range(n_layers):

        for i in range(bs[l_no].shape[0]):

            bs[l_no][i] -= h

            _,cost1 = compute_cost(X, Y, Ws, bs, gammas, betas,lamda)

            bs[l_no][i] += 2*h
            _, cost2 = compute_cost(X, Y, Ws, bs, gammas, betas,lamda)
            grad_bs[l_no][i] = (cost2 - cost1) / (2*h)
            bs[l_no][i] -= h

        for i in range(Ws[l_no].shape[0]):
            for j in range(Ws[l_no].shape[1]):

                Ws[l_no][i, j] -= h
                _, cost1 = compute_cost(X, Y, Ws, bs, gammas, betas,lamda)
                Ws[l_no][i, j] += 2*h
                _, cost2 = compute_cost(X, Y, Ws, bs, gammas, betas,lamda)
                grad_ws[l_no][i,j] = (cost2 - cost1) / (2*h)
                Ws[l_no][i, j] -= h

        if l_no<n_layers-1:

            for i in range(gammas[l_no].shape[0]):

                gammas[l_no][i] -= h

                _,cost1 = compute_cost(X, Y, Ws, bs, gammas, betas,lamda)

                gammas[l_no][i] += 2*h
                _, cost2 = compute_cost(X, Y, Ws, bs, gammas, betas,lamda)
                grad_gammas[l_no][i] = (cost2 - cost1) / (2*h)
                gammas[l_no][i] -= h

            for i in range(betas[l_no].shape[0]):

                betas[l_no][i] -= h

                _,cost1 = compute_cost(X, Y, Ws, bs, gammas, betas,lamda)

                betas[l_no][i] += 2*h
                _, cost2 = compute_cost(X, Y, Ws, bs, gammas, betas,lamda)
                grad_betas[l_no][i] = (cost2 - cost1) / (2*h)
                betas[l_no][i] -= h

    return grad_bs, grad_ws, grad_gammas, grad_betas





def check_gradients():
    X, Y, y = load_data("data_batch_1")
    Ws, bs, gammas, betas = initials()

    X_reduced = X[:10, 0:2]
    Y_reduced = Y[:, 0:2]
    W_reduced = []
    W_reduced.append(Ws[0][:, :10])
    for layer in range(1, n_layers):
        W_reduced.append(Ws[layer])

    grad_bAnalytic, grad_WAnalytic,grad_gammas_a,grad_betas_a = compute_grad_analytic(X_reduced, Y_reduced,W_reduced, bs, gammas,betas,lamda)

    grad_bNumeric, grad_WNumeric, grad_gammas_n, grad_betas_n = compute_grad_numeric(X_reduced, Y_reduced,W_reduced, bs, gammas, betas,lamda)

    for layer in range(n_layers):
        grad_w_a, grad_w_n = grad_WAnalytic[layer], grad_WNumeric[layer]
        grad_b_a, grad_b_n = grad_bAnalytic[layer], grad_bNumeric[layer]
        print("layer", layer)
        print("------")
        print("grad(W):     ","Mean absolute diff.: %.2e" % np.mean(np.abs(grad_w_a - grad_w_n)))
        print("grad(b):     ","Mean absolute diff.: %.2e" % np.mean(np.abs(grad_b_a - grad_b_n)))

        if (layer < n_layers - 1) and batch_norm:
            grad_gamma_a, grad_gamma_n = grad_gammas_a[layer], grad_gammas_n[layer]
            grad_beta_a, grad_beta_n = grad_betas_a[layer], grad_betas_n[layer]
            print("grad(gamma): ","Mean absolute diff.: %.2e" % np.mean(np.abs(grad_gamma_a - grad_gamma_n)))
            print("grad(beta):  ","Mean absolute diff.: %.2e" % np.mean(np.abs(grad_beta_a - grad_beta_n)),"\n")
    print()


def fit(X, Y, X_val, Y_val, y_val, lamda):

    W, b, gamma, beta = initials()
    cost_values = []
    cost_val_values = []
    accuracy_val_values = []
    iterations =  []

    n_s = 5 * int(X.shape[1]/ batch_size)
    totIters = int(2* n_cycles*n_s)
    numBatches = int(X.shape[1] /batch_size)
    n_epochs = int (totIters / numBatches)
    lrs=[]
    eta=lr_min
    for epoch in range(n_epochs):
        print("Epoch: {}/{}".format(epoch,n_epochs))
        X , Y = shuffle(X, Y)

        for j in range( numBatches ):
            lrs.append(eta)
            j_start = j * batch_size
            j_end = j_start + batch_size
            X_batch = X[:, j_start: j_end]
            Y_batch = Y[:, j_start:j_end]

            grad_b , grad_W , grad_gamma, grad_beta = compute_grad_analytic(X_batch , Y_batch  ,W ,b , gamma , beta,lamda)

            for layer in range(n_layers):
                W[layer] = W[layer] - (eta * grad_W[layer])
                b[layer] = b[layer] - (eta * grad_b[layer])

                if batch_norm and layer != n_layers-1 :
                    gamma[layer] = gamma[layer] - (eta * grad_gamma[layer])
                    beta[layer] = beta[layer] - (eta * grad_beta[layer])
            #print(np.mean(grad_W[0]),np.mean(grad_W[1]),np.mean(grad_b[0]),np.mean(grad_b[1]))
            #update iteration info
            upd = epoch * numBatches + j
            eta = cyclic_lr(n_s, upd)

        loss, cost = compute_cost(X, Y, W , b, gamma, beta,lamda)

        print(np.mean(W[0]))
        cost_values.append(cost)
        # on validation data
        activationsVal, probabilitiesVal, predictionsVal , scoresVal , sHatsVal, meansVal , variancesVal = evaluate_classifier(X_val,  W, b, gamma, beta)
        lossVal , costVal = compute_cost(X_val, Y_val, W , b, gamma, beta,lamda)
        accuracyVal = compute_accuracy(predictionsVal , y_val)
        cost_val_values.append(costVal)
        accuracy_val_values.append(accuracyVal)
        print('Loss,Cost,AccVal:', loss, cost,accuracyVal)
        iterations.append(upd)

    # On test data

    return lrs,W,b,gamma,beta,iterations, cost_values , cost_val_values , accuracy_val_values


def predict(Xtest, W_l, b_l, gammas,betas):
    activations = []
    scores = []
    s_hs = []

    for l_no in range(n_layers-1):
        if l_no==0:
            score=np.dot(W_l[l_no], Xtest) + b_l[l_no]
            scores.append(score)
        else:
            score = np.dot(W_l[l_no], activations[l_no-1]) + b_l[l_no]
            scores.append(score)

        if batch_norm:
            s_h = batch_normalize(score,mu_avg[l_no],var_avg[l_no])
            #means.append(mean)
            #variances.append(variance)
            s_hs.append(s_h)
            s_t = np.multiply(gammas[l_no], s_h) + betas[l_no]
            activations.append(np.maximum(0, s_t))

        else:
            activations.append(np.maximum(0, scores[l_no]))


    #print(len(W_l),len(activations),len(b_l),n_layers)
    S = np.dot(W_l[n_layers-1], activations[n_layers-2]) + b_l[n_layers-1]
    scores.append(S)


    probas = softmax(S)
    preds = np.argmax(probas, axis=0)


    return probas, preds


check_gradients()


# #course search
# lamda_values = [7e-3]
# lmd_acc = []
#
# a=0
# for lmd in lamda_values:
#
#     a+=1
#     lamda=lmd
#     print('{}/{}'.format(a,len(lamda_values)),lamda)
#
#     lrs, W_l, b_l, gamma, beta, iters, cost_values, cost_val_values, accuracy_val_values = fit(X, Y, X_val, Y_val, y_val,lamda)
#
#     lmd_acc.append((np.max(accuracy_val_values),lmd))
#     mu_avg, var_avg = None, None
# best_lamda = max(lmd_acc)
# print('Best lambda is',best_lamda[1],'with accuracy:',best_lamda[0])


#lamda=7e-3

lrs,W_l, b_l, gamma, beta, iters,  cost_values, cost_val_values, accuracy_val_values = fit(X, Y, X_val, Y_val, y_val, lamda)

var_avg = np.array(var_avg)

probabilitiesTest, predictionsTest = predict(X_test, W_l, b_l, gamma, beta)

testAccuracy = compute_accuracy(predictionsTest, y_test)
print("\n")
print("Test accuracy:", testAccuracy, "\n")

plt.figure()
plt.plot(iters, cost_values)
plt.plot(iters, cost_val_values)
plt.legend(['Training','Validation'],loc='best')
plt.title('Cost over update')
plt.xlabel('Update step')
plt.ylabel('Cost')
plt.savefig('cost_{}_{}_{}_{}_{}.png'.format(n_layers,n_cycles,lamda,batch_norm,siggma))


plt.figure()
plt.plot(iters,accuracy_val_values)

plt.legend(['Validation'],loc='best')
plt.title('Accuracy over update')
plt.xlabel('Update step')
plt.ylabel('Accuracy')
plt.ylim(0,)
plt.savefig('accuracy_{}_{}_{}_{}_{}.png'.format(n_layers,n_cycles,lamda,batch_norm,siggma))
plt.show()
