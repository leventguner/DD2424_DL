import numpy as np
import matplotlib.pyplot as plt
import pickle

from math import sqrt
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

    mean_s = np.mean(X,axis=0)
    std_d = np.std(X, axis=0)
    X=X-mean_s
    X=X/std_d

    y = np.array(data[b'labels'])
    N = X.shape[1]
    k = int(max(y)+1)
    Y = np.zeros((k, N))
    Y[y, np.arange(N)] = 1
    return X, Y, y

Xa,Ya,ya = load_data("data_batch_1")


for i in [2,3,4,5]:
    Xb, Yb, yb = load_data("data_batch_{}".format(i))
    Xa = np.concatenate([Xa, Xb],axis=1)
    Ya = np.concatenate([Ya, Yb],axis=1)
    ya = np.concatenate([ya, yb])

val_size=1000
X,Y,y = Xa[:,:-val_size],Ya[:,:-val_size],ya[:-val_size]
X_val, Y_val, y_val = Xa[:,-val_size:],Ya[:,-val_size:],ya[-val_size:]
print(X.shape,Y.shape,y.shape)
print(X_val.shape,Y_val.shape,y_val.shape)

XTest, YTest, yTest = load_data("test_batch")

N = X.shape[1]
k = int(max(y)+1)
d = X.shape[0]

parameters = {
    'mu': 0,
    'h_size':50,
    'h': 1e-5,
    'batch_size': 100,
    'n_epoch':14,
    'lamda': 8e-3,
    'lr_min': 1e-5,
    'lr_max':1e-1,
    'l_min':-7,
    'l_max':-2,
    'cycles':2,
    'n_lambda':10,
    'n_s':900
}


mu = parameters['mu']
m = parameters['h_size']
h = parameters['h']
batch_size = parameters['batch_size']
n_epochs = parameters['n_epoch']
lamda = parameters['lamda']
sigma1=1/sqrt(d)
sigma2=1/sqrt(m)
lr_min = parameters['lr_min']
lr_max = parameters['lr_max']
l_min = parameters['l_min']
l_max = parameters['l_max']
n_lambda = parameters['n_lambda']
n_cycles = parameters['cycles']
n_s = parameters['n_s']



def initials():
    """
    Define initial W and b
    """
    W1 = np.random.normal(mu, sigma1, (m, d))
    W2 = np.random.normal(mu, sigma2, (k, m))
    b1 = np.zeros((m, 1))
    b2 = np.zeros((k, 1))

    return W1,W2,b1,b2

W1,W2,b1,b2 = initials()

def cyclic_lr(n_s , upd ,lr_min=lr_min,lr_max=lr_max):

    lr_g = int(upd/(2*n_s))
    diff = lr_max-lr_min
    if upd < 2*lr_g*n_s + n_s:
        lr = lr_min + (upd-2*lr_g*n_s)/n_s*diff

    else:
        lr = lr_max - (upd-(2*lr_g+1)*n_s)/n_s*diff

    return lr

def cyclic_lambda(l_min=l_min,l_max=l_max):
    l = l_min + (l_max-l_min) * np.random.rand()
    lamda = 10**l

    return lamda


def softmax(S):
    """
    Input:Wx+b / Output:Softmax
    """
    soft = np.exp(S) / np.sum(np.exp(S), axis=0)
    return soft


def evaluate_classifier(X, W1=W1, W2=W2, b1=b1, b2=b2):
    """
    Evaluate probabilities with softmax
    """

    S1 = np.dot(W1 , X) + b1

    activations = np.maximum(0, S1)
    S = np.dot(W2, activations) + b2
    probas = softmax(S)
    return probas,activations


def compute_cost(X, Y, W1, W2, b1, b2):
    """
    Compute the cost
    """
    pred,act = evaluate_classifier(X, W1, W2, b1, b2)
    num_e = X.shape[1]
    L2reg = lamda * (np.sum(W1 ** 2) + np.sum(W2 ** 2))
    y_p = np.multiply(Y, pred).sum(axis=0)
    cross_ent = np.sum(-np.log(y_p))
    #
    loss = cross_ent / num_e
    cost = loss + L2reg

    return loss,cost


def compute_accuracy(predictions, y):
    """
    Compute the accuracy
    """

    corr = len(np.where(predictions == y)[0])
    accuracy = (corr / len(y))
    return accuracy


def compute_grad_analytic(X, Y, W1, W2, b1, b2):
    """
    Compute the gradient analytically
    """

    probas,acts = evaluate_classifier(X, W1, W2, b1, b2)
    indicator = 1 * (acts > 0)
    ones = np.ones((X.shape[1], 1))
    grad = - (Y - probas)
    grad_b2 = np.dot(grad, ones) / X.shape[1]

    grad_W2 = np.dot(grad, acts.T) / X.shape[1]
    grad_W2 = np.add(grad_W2, (2 * lamda * W2))

    grad = np.dot(W2.T , grad)
    grad = np.multiply(grad, indicator)

    grad_b1 = np.dot(grad, ones) / X.shape[1]

    grad_W1 = np.dot(grad, X.T) / X.shape[1]
    grad_W1 = np.add(grad_W1, (2 * lamda * W1))

    return grad_W1, grad_W2, grad_b1, grad_b2


def compute_grad_numeric(X, Y, W1, W2, b1, b2):
    """
    Compute the gradient numerically
    """

    grad_W1 = np.zeros((W1.shape[0], W1.shape[1]))
    grad_W2 = np.zeros((W2.shape[0], W2.shape[1]))
    grad_b1 = np.zeros((m, 1))
    grad_b2 = np.zeros((k, 1))

    for i in range(b1.shape[0]):
        b1[i] -= h

        _,cost1 = compute_cost(X, Y, W1, W2, b1, b2)
        b1[i] += 2 * h
        _,cost2 = compute_cost(X, Y, W1, W2, b1, b2)
        grad_b1[i] = (cost2 - cost1) / (2 * h)
        b1[i] -= h

    for i in range(b2.shape[0]):
        b2[i] -= h

        _,cost1 = compute_cost(X, Y, W1, W2, b1, b2)
        b2[i] += 2 * h
        _,cost2 = compute_cost(X, Y, W1, W2, b1, b2)
        grad_b2[i] = (cost2 - cost1) / (2 * h)
        b2[i] -= h


    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            W1[i, j] -= h
            _,cost1 = compute_cost(X, Y, W1, W2, b1, b2)
            W1[i, j] += 2 * h

            _,cost2 = compute_cost(X, Y, W1, W2, b1, b2)
            grad_W1[i, j] = (cost2 - cost1) / (2 * h)
            W1[i, j] -= h

    for i in range(W2.shape[0]):
        for j in range(W2.shape[1]):
            W2[i, j] -= h
            _,cost1 = compute_cost(X, Y, W1, W2, b1, b2)
            W2[i, j] += 2 * h

            _,cost2 = compute_cost(X, Y, W1, W2, b1, b2)
            grad_W2[i, j] = (cost2 - cost1) / (2 * h)
            W2[i, j] -= h

    return grad_W1, grad_W2, grad_b1, grad_b2


def check_gradients():
    X, Y, y = load_data("data_batch_1")
    W1, W2 , b1 , b2 = initials()
    grad_w1_a , grad_w2_a, grad_b1_a , grad_b2_a  = compute_grad_analytic(X[:20 , 0:1], Y[:, 0:1] , W1[:, :20] , W2 , b1, b2 )
    grad_w1_n , grad_w2_n, grad_b1_n , grad_b2_n = compute_grad_numeric(X[:20, 0:1], Y[:, 0:1] , W1[:, :20] , W2 , b1, b2 )

    print("grad(W1)")
    print("--------")
    print('Analytic grad(W1): {}'.format(np.sum(grad_w1_a)))
    print('Numerical grad(W1): {}'.format(np.sum(grad_w1_n)))
    print('Sum of absolute diff.: %.2e' %np.sum(np.abs(grad_w1_a - grad_w1_n)))

    print("\ngrad(W2)")
    print("--------")
    print('Analytic grad(W2): {}'.format(np.sum(grad_w2_a)))
    print('Numerical grad(W2): {}'.format(np.sum(grad_w2_n)))
    print('Sum of absolute diff.: %.2e' %np.sum(np.abs(grad_w2_a - grad_w2_n)))

    print("\ngrad(b1)")
    print("--------")
    print('Analytic grad(b1): {}'.format(np.sum(grad_b1_a)))
    print('Numerical grad(b1): {}'.format(np.sum(grad_b1_n)))
    print('Sum of absolute diff.: %.2e' %np.sum( np.abs(grad_b1_a - grad_b1_n)))

    print("\ngrad(b2)")
    print("--------")
    print('Analytic grad(b2): {}'.format(np.sum(grad_b2_a)))
    print('Numerical grad(b2): {}'.format(np.sum(grad_b2_n)))
    print('Sum of absolute diff.: %.2e' %np.sum(np.abs(grad_b2_a - grad_b2_n)))

def fit(lmd,X, Y, y, X_val, Y_val, y_val, W1=W1, W2=W2 , b1=b1, b2=b2):
    global lamda
    lamda = lmd

    #Store results
    accuracy_values = []
    accuracy_val_values = []
    cost_values = []
    cost_val_values = []
    loss_values = []
    loss_val_values = []
    iterations =  []
    lrs = []

    n_s = 2*int(N /batch_size)
    print('stepsize',n_s)
    tot_iters = int(2* n_cycles*n_s)
    num_batches = int(N / batch_size)
    n_epochs = int(tot_iters/num_batches)


    lr=lr_min
    upd=0

    for epoch in range(n_epochs):
        print("Epoch: {}/{}".format(epoch,n_epochs))
        for j in range(num_batches):
            lrs.append(lr)
            j_start = j*batch_size
            j_end = j_start + batch_size
            X_batch = X[: ,j_start: j_end]
            Y_batch  = Y[: ,j_start:j_end]

            grad_W1 , grad_W2 , grad_b1 , grad_b2 = compute_grad_analytic(X_batch , Y_batch  ,W1,W2, b1 , b2)
            W1 = W1 - (lr * grad_W1)
            b1 = b1 - (lr * grad_b1)
            W2 = W2 - (lr * grad_W2 )
            b2 = b2 - (lr * grad_b2)
            print(np.mean(grad_W1),np.mean(grad_W2),np.mean(grad_b1),np.mean(grad_b2))
            upd = epoch*num_batches+j


            lr = cyclic_lr(n_s, upd)



        probas,acts = evaluate_classifier(X, W1, W2, b1, b2)
        loss, cost = compute_cost(X, Y, W1, W2, b1, b2)

        predictions = np.argmax(probas, axis=0)
        accuracy = compute_accuracy(predictions , y)
        cost_values.append(cost)
        loss_values.append(loss)
        accuracy_values.append(accuracy)
        
        #validation

        probas_val, activations_val = evaluate_classifier(X_val , W1, W2 , b1 , b2)
        loss_val , cost_val = compute_cost(X_val, Y_val, W1 , W2,b1,b2)
        predictions_val = np.argmax(probas_val, axis=0)
        accuracy_val = compute_accuracy(predictions_val , y_val)
        cost_val_values.append(cost_val)
        loss_val_values.append(loss_val)
        accuracy_val_values.append(accuracy_val)
        
        iterations.append(upd)

    return  W1,W2,b1,b2, iterations, lrs, loss_values , loss_val_values, accuracy_values  , accuracy_val_values , cost_values , cost_val_values


check_gradients()

def predict(X,W1=W1,W2=W2,b1=b1,b2=b2):
    """
    Evaluate probabilities with softmax
    """

    S1 = np.dot(W1 , X) + b1
    activations = np.maximum(0, S1)
    S = np.dot(W2, activations) + b2
    probas = softmax(S)
    preds = np.argmax(probas, axis=0)
    return probas,preds


#course search
# lamda_values = []
# lmd_acc = []
# for i in range(n_lambda):
#     lmds = cyclic_lambda()
#     lamda_values.append(lmds)
# a=0
# for lmds in lamda_values:
#     a+=1
#     print('{}/{}'.format(a,len(lamda_values)))
#
#     W1, W2 , b1, b2 = initials()
#
#     _,_,_,_, iterations, lrs, loss_values, loss_val_values, accuracy_values, accuracy_val_values, cost_values, cost_val_values = fit(
#         lmds, X, Y, y, X_val, Y_val, y_val)
#
#     lmd_acc.append((np.max(accuracy_val_values),lmds))
#
# best_lamda = max(lmd_acc)
# print('Best lambda is',best_lamda[1],'with accuracy:',best_lamda[0])



#fine search
# lmd_acc = []
# for lmds in np.linspace(0,best_lamda[1],10):
#     a+=1
#     print('{}/{}'.format(a,len(lamda_values)))
#
#     W1, W2 , b1, b2 = initials()
#     _,_,_,_, iterations, lrs, loss_values, loss_val_values, accuracy_values, accuracy_val_values, cost_values, cost_val_values = fit(
#         lmds, X, Y, y, X_val, Y_val, y_val)
#
#     lmd_acc.append((np.max(accuracy_val_values),lmds))



lmds = 1.17e-7

W1,W2,b1,b2 = initials()
W1,W2,b1,b2, iterations, lrs, loss_values , loss_val_values, accuracy_values, accuracy_val_values , cost_values , cost_val_values= fit(lmds,X, Y, y, X_val, Y_val, y_val)

probasTest, predsTest = predict(XTest,W1,W2,b1,b2)

testAccuracy = compute_accuracy(predsTest, yTest)

plt.figure()
plt.plot(np.arange(len(lrs)),lrs)
plt.legend(['min eta=1e-5, max eta=1e-1'],loc='best')
plt.title('LR over update')
plt.xlabel('Update step')
plt.ylabel('Learning rate')
plt.savefig('lr_{}_{}_{}.png'.format(n_cycles,n_s,lamda))


plt.figure()
plt.plot(iterations,cost_values)
plt.plot(iterations,cost_val_values)
plt.legend(['Training','Validation'],loc='best')
plt.title('Cost over update')
plt.xlabel('Update step')
plt.ylabel('Cost')
plt.ylim(0,)
plt.savefig('cost_{}_{}_{}.png'.format(n_cycles,n_s,lamda))


plt.figure()
plt.plot(iterations,loss_values)
plt.plot(iterations,loss_val_values)
plt.legend(['Training','Validation'],loc='best')
plt.title('Loss over update')
plt.xlabel('Update step')
plt.ylabel('Loss')
plt.ylim(0,)
plt.savefig('loss_{}_{}_{}.png'.format(n_cycles,n_s,lamda))


plt.figure()
plt.plot(iterations,accuracy_values)
plt.plot(iterations,accuracy_val_values)
plt.legend(['Training','Validation'],loc='best')
plt.title('Accuracy over update')
plt.xlabel('Update step')
plt.ylabel('Accuracy')
plt.ylim(0,)
plt.savefig('accuracy_{}_{}_{}.png'.format(n_cycles,n_s,lamda))
plt.show()

print("\n" )
print("Val accuracy: ", np.max(accuracy_val_values), "\n" )
print("Test accuracy: ", testAccuracy, "\n" )