import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

m = 100
eta = 0.1
seq_length = 25
h=1e-4
mu = 0
sigma = 0.01
eps = 1e-8

def get_data():
    
    global k
    data = open("goblet_book.txt", "r").read()
    data = list(data)
    chars = list(set(data))
    return data, chars

_,charss=get_data()
k = len(charss)


def initials():
    parameters = {}
    parameters['b'] = np.zeros((m, 1))
    parameters['c'] = np.zeros((k, 1))
    parameters['u'] = np.random.normal(mu, sigma, (m, k))
    parameters['h'] = np.zeros((m, 1))
    parameters['v'] = np.random.normal(mu, sigma, (k, m))
    parameters['w'] = np.random.normal(mu, sigma, (m, m))
    return parameters


def char_to_ind(chars):
    char_ind_dict = {chars[i]: i for i in range(len(chars))}
    return char_ind_dict

def ind_to_char(char_ind_dict):
    ind_char_dict = {v: k for k, v in char_ind_dict.items()}
    return ind_char_dict

def char_to_onehot(char_ind_dict, book_data):
    one_hot = np.zeros((len(char_ind_dict), len(book_data)))
    inds = [[char_ind_dict[char],i] for i,char in enumerate(book_data)]

    for i,j in inds:
        one_hot[i,j]=1

    return one_hot

def onehot_to_char(y, ind_char_dict):
    inds = [np.where(r == 1)[0][0] for r in y]
    seqs = [ind_char_dict[ind] for ind in inds]
    sequence = ''.join(seqs)
    return sequence

def generate(b, c, h, u, v, w, x0, n):
    ys = []
    x = x0
    H = [h]
    for i in range(n):
        a = np.dot(w, H[i]) + np.dot(u, x) + b
        H.append(np.tanh(a))
        p = softmax(np.dot(v, H[-1]) + c)
        i_sel = get_ind_from_prob(p)
        x = np.zeros((k,1))
        x[i_sel][0] = 1
        ys.append(x)
    return ys

def get_ind_from_prob(p):
    cp = np.cumsum(p, axis=0)
    r = np.random.rand()
    cp = cp - r
    i_s = np.where(cp > 0)
    i_sel = i_s[0][0]
    return i_sel


def softmax(S):
    """
    Input:Wx+b / Output:Softmax
    """
    soft = np.exp(S) / np.sum(np.exp(S), axis=0)
    return soft

def evaluate_classifier(X, Y, b, c, h, u, v, w):
    P = []
    H = [0]*X.shape[1]
    H.append(h)
    loss = 0
    
    for t in range(X.shape[1]):
        X_t = X[:, t].reshape(X.shape[0], 1)
        at = np.dot(w, H[t-1]) + np.dot(u, X_t) + b
        H[t] = np.tanh(at)
        p_a = softmax(np.dot(v, H[t]) + c)
        P.append(p_a)
        loss -= np.log(np.dot(Y[:, t].T, p_a))
        
    return loss, P, H


def compute_grad_analytic(X, Y, b, c, h, u, v, w):
    loss, P, H = evaluate_classifier(X, Y, b, c, h, u, v, w)
    grad_b = np.zeros((m,1))
    grad_c = np.zeros((k,1))
    grad_u = np.zeros((m,k))
    grad_w = np.zeros((m,m))
    grad_v = np.zeros((k,m))
    grad_a = np.zeros((m,1))
    gs  = []

    for t in range(X.shape[1]):
        Y_t = Y[:, t].reshape(Y.shape[0], 1)
        gs.append(-(Y_t - P[t]))
    gs = np.array(gs)

    for t in reversed(range(X.shape[1])):
        Xt = X[:, t].reshape(X.shape[0], 1)
        grad_c += gs[t]
        grad_v += np.dot(gs[t], H[t].T)

        dh = np.dot(v.T, gs[t]) + np.dot(w.T, grad_a)

        grad_a = np.multiply(dh, (1 - (H[t] ** 2)))
        grad_w += np.dot(grad_a, H[t - 1].T)
        grad_b += grad_a
        grad_u += np.dot(grad_a, Xt.T)


    return grad_b, grad_c, H[-1], grad_u, grad_v, grad_w, loss


def compute_grad_numeric(X, Y, b, c, h0,  u, v, w):
    grad_b = np.zeros(b.shape)
    grad_c = np.zeros(c.shape)
    grad_u = np.zeros(u.shape)
    grad_w = np.zeros(w.shape)
    grad_v = np.zeros(v.shape)
    for i in range(len(b)):
        b[i] -= h
        c1,_,_ = evaluate_classifier(X, Y, b, c, h0, u, v, w)

        b[i] += 2*h
        c2,_,_ = evaluate_classifier(X, Y, b, c, h0, u, v, w)

        grad_b[i] = (c2 - c1) / (2*h)
        b[i] -= h
    for i in range(len(c)):
        c[i] -= h
        c1,_,_ = evaluate_classifier(X, Y, b, c, h0, u, v, w)

        c[i] += 2*h
        c2,_,_ = evaluate_classifier(X, Y, b, c, h0, u, v, w)

        grad_c[i] = (c2 - c1) / (2*h)
        c[i] -= h
    for i in range(u.shape[0]):
        for j in range(u.shape[1]):
            u[i][j] -= h
            c1,_,_ = evaluate_classifier(X, Y, b, c, h0, u, v, w)

            u[i][j] += 2 * h
            c2,_,_ = evaluate_classifier(X, Y, b, c, h0, u, v, w)

            grad_u[i][j] = (c2 - c1) / (2 * h)
            u[i][j] -= h

    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            w[i][j] -= h
            c1,_,_ = evaluate_classifier(X, Y, b, c, h0, u, v, w)

            w[i][j] += 2 * h
            c2,_,_ = evaluate_classifier(X, Y, b, c, h0, u, v, w)

            grad_w[i][j] = (c2 - c1) / (2 * h)
            w[i][j] -= h
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            v[i][j] -= h
            c1,_,_ = evaluate_classifier(X, Y, b, c, h0, u, v, w)

            v[i][j] += 2 * h
            c2,_,_ = evaluate_classifier(X, Y, b, c, h0, u, v, w)

            grad_v[i][j] = (c2 - c1) / (2 * h)
            v[i][j] -= h

    return grad_b, grad_c, grad_u, grad_v, grad_w


def check_gradients():
    book_data, chars = get_data()
    char_ind_dict = char_to_ind(chars)

    params = initials()
    b, c, u, h0, v, w = params.values()

    X = char_to_onehot(char_ind_dict, book_data[0:seq_length])
    Y = char_to_onehot(char_ind_dict, book_data[1:seq_length + 1])

    gb_a, gc_a, _, gu_a ,gv_a ,gw_a,_ = compute_grad_analytic(X, Y, b, c, h0, u, v, w)
    gb_n, gc_n, gu_n, gv_n, gw_n = compute_grad_numeric(X, Y, b, c, h0,  u, v, w)

    print("grad(b)")
    print("-------")
    print("Mean absolute diff.: %.2e" %np.mean(np.abs(gb_a - gb_n)), "\n")
    print("grad(c)")
    print("-------")
    print("Mean absolute diff.: %.2e" %np.mean(np.abs(gc_a - gc_n)), "\n")
    print("grad(U)")
    print("-------")
    print("Mean absolute diff.: %.2e" %np.mean(np.abs(gu_a - gu_n)), "\n")
    print("grad(V)")
    print("-------")
    print("Mean absolute diff.: %.2e" %np.mean(np.abs(gv_a - gv_n)), "\n")
    print("grad(W)")
    print("-------")
    print("Mean absolute diff.: %.2e" %np.mean(np.abs(gw_a - gw_n)), "\n")

def train(max_iter_n):
    book_data, chars = get_data()
    char_ind_dict = char_to_ind(chars)
    ind_char_dict = ind_to_char(char_ind_dict)

    params = initials()
    b, c, u, h_start, v, w = params.values()

    h = h_start
    
    Us = []
    Vs = []
    Ws = []
    bs = []
    cs = []

    mb = np.zeros(b.shape)
    mc = np.zeros(c.shape)
    mU = np.zeros(u.shape)
    mW = np.zeros(w.shape)
    mV = np.zeros(v.shape)
    
    e = 0
    iteration = 0
    smooth_loss = -np.log(1 / k) * seq_length
    smooth_losses = []
    while (iteration < max_iter_n):
        if e >= len(book_data) - seq_length - 1:
            e = 0
            h = h_start

        X_chars = book_data[e: e + seq_length]
        Y_chars = book_data[e + 1: e + 1 + seq_length]

        X = char_to_onehot(char_ind_dict, X_chars)
        Y = char_to_onehot(char_ind_dict, Y_chars)

        grad_b, grad_c, hR, grad_u, grad_v, grad_w, loss = compute_grad_analytic(X, Y, b, c, h, u, v, w)
        h = hR
        smooth_loss = (0.999 * smooth_loss) + (0.001 * loss)
        smooth_losses.append(smooth_loss)

        if (iteration % 10000 == 0):
            print("Generated text at {}th iteration:".format(iteration),"\n")
            y = generate(b, c, h, u, v, w, X[:, 0], n=200)
            text = onehot_to_char(y, ind_char_dict)
            print(text, "\n\n")
        # AdaGrad update

        mW += grad_w**2
        w += - (eta * grad_w) / np.sqrt(mW + eps)
        Ws.append(w)

        mV += grad_v**2
        v += - (eta * grad_v) / np.sqrt(mV + eps)
        Vs.append(v)

        mU += grad_u ** 2
        u += - (eta * grad_u) / np.sqrt(mU + eps)
        Us.append(u)

        mb += grad_b ** 2
        b += - (eta * grad_b) / np.sqrt(mb + eps)
        bs.append(b)

        mc += grad_c ** 2
        c += - (eta * grad_c) / np.sqrt(mc + eps)
        cs.append(c)

        e += seq_length
        iteration += 1
    best = np.argmin(smooth_losses)
    
    bestU = Us[best]
    bestV = Vs[best]
    bestW = Ws[best]
    bestb = bs[best]
    bestc = cs[best]

    return smooth_losses, bestb, bestc, h, bestU, bestV, bestW, X[:,0], ind_char_dict




check_gradients()

max_iter_n = 100000
smooth_losses, bestb, bestc, besth, bestU, bestV, bestW, X, ind_char_dict = train(max_iter_n)
smooth_losses = [i[0] for i in smooth_losses]


plt.figure()
plt.plot(np.arange(len(smooth_losses)), smooth_losses)
plt.xlabel('Iteration')
plt.ylabel('Smooth Loss')
plt.savefig('{}.png'.format(max_iter_n))

print("Generated text with best model")
print("------------------------------")
for i in range(10):
    y_best = generate(bestb, bestc, besth, bestU, bestV, bestW, X, n=1000)
    text_best = onehot_to_char(y_best, ind_char_dict)
    print(text_best)
    print("------------------------------")

