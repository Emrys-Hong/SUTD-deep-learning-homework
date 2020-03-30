from builtins import range
import numpy as np

####x
# def affine_forward(x, w, b):
#     """
#     Computes the forward pass for an affine (fully-connected) layer.

#     The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
#     examples, where each example x[i] has shape (d_1, ..., d_k). We will
#     reshape each input into a vector of dimension D = d_1 * ... * d_k, and
#     then transform it to an output vector of dimension M.

#     Inputs:
#     - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
#     - w: A numpy array of weights, of shape (D, M)
#     - b: A numpy array of biases, of shape (M,)

#     Returns a tuple of:
#     - out: output, of shape (N, M)
#     - cache: (x, w, b)
#     """
#     out = None
#     ###########################################################################
#     # TODO: Implement the affine forward pass. Store the result in out. You   #
#     # will need to reshape the input into rows.                               #
#     ###########################################################################
#     # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

#     N = x.shape[0]
#     D, M = w.shape
#     out = x.reshape(N, D) @ w + b

#     # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
#     ###########################################################################
#     #                             END OF YOUR CODE                            #
#     ###########################################################################
#     cache = (x, w, b)
#     return out, cache


# ####x
# def affine_backward(dout, cache):
#     """
#     Computes the backward pass for an affine layer.

#     Inputs:
#     - dout: Upstream derivative, of shape (N, M)
#     - cache: Tuple of:
#       - x: Input data, of shape (N, d_1, ... d_k)
#       - w: Weights, of shape (D, M)
#       - b: Biases, of shape (M,)

#     Returns a tuple of:
#     - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
#     - dw: Gradient with respect to w, of shape (D, M)
#     - db: Gradient with respect to b, of shape (M,)
#     """
#     x, w, b = cache
#     dx, dw, db = None, None, None
#     ###########################################################################
#     # TODO: Implement the affine backward pass.                               #
#     ###########################################################################
#     # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

#     D,M = w.shape
#     N = x.shape[0]
#     db = np.sum(dout,axis = 0)
    
#     X_ravel = x.reshape((N,D))
#     dw = X_ravel.T.dot(dout)
    
#     dX_ravel = dout.dot(w.T)
#     dx = dX_ravel.reshape(x.shape)

#     # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
#     ###########################################################################
#     #                             END OF YOUR CODE                            #
#     ###########################################################################
#     return dx, dw, db


# ####x
# def relu_forward(x):
#     """
#     Computes the forward pass for a layer of rectified linear units (ReLUs).

#     Input:
#     - x: Inputs, of any shape

#     Returns a tuple of:
#     - out: Output, of the same shape as x
#     - cache: x
#     """
#     out = None
#     ###########################################################################
#     # TODO: Implement the ReLU forward pass.                                  #
#     ###########################################################################
#     # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


#     out = x
#     out[out<0] = 0
    
#     # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
#     ###########################################################################
#     #                             END OF YOUR CODE                            #
#     ###########################################################################
#     cache = x
#     return out, cache


# ####x
# def relu_backward(dout, cache):
#     """
#     Computes the backward pass for a layer of rectified linear units (ReLUs).

#     Input:
#     - dout: Upstream derivatives, of any shape
#     - cache: Input x, of same shape as dout

#     Returns:
#     - dx: Gradient with respect to x
#     """
#     dx, x = None, cache
#     ###########################################################################
#     # TODO: Implement the ReLU backward pass.                                 #
#     ###########################################################################
#     # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

#     dx = np.ones(x.shape)
#     dx = dx*dout
#     dx[x<=0] = 0

#     # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
#     ###########################################################################
#     #                             END OF YOUR CODE                            #
#     ###########################################################################
#     return dx





# ###x
# def dropout_forward(x, dropout_param):
#     """
#     Performs the forward pass for (inverted) dropout.

#     Inputs:
#     - x: Input data, of any shape
#     - dropout_param: A dictionary with the following keys:
#       - p: Dropout parameter. We keep each neuron output with probability p.
#       - mode: 'test' or 'train'. If the mode is train, then perform dropout;
#         if the mode is test, then just return the input.
#       - seed: Seed for the random number generator. Passing seed makes this
#         function deterministic, which is needed for gradient checking but not
#         in real networks.

#     Outputs:
#     - out: Array of the same shape as x.
#     - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
#       mask that was used to multiply the input; in test mode, mask is None.

#     NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
#     See http://cs231n.github.io/neural-networks-2/#reg for more details.

#     NOTE 2: Keep in mind that p is the probability of **keep** a neuron
#     output; this might be contrary to some sources, where it is referred to
#     as the probability of dropping a neuron output.
#     """
#     p, mode = dropout_param['p'], dropout_param['mode']
#     if 'seed' in dropout_param:
#         np.random.seed(dropout_param['seed'])

#     mask = None
#     out = None

#     if mode == 'train':
#         #######################################################################
#         # TODO: Implement training phase forward pass for inverted dropout.   #
#         # Store the dropout mask in the mask variable.                        #
#         #######################################################################
#         # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

#         p = dropout_param['p']
#         mask = (np.random.rand(*x.shape) < p) / p
#         out = x*mask

#         # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
#         #######################################################################
#         #                           END OF YOUR CODE                          #
#         #######################################################################
#     elif mode == 'test':
#         #######################################################################
#         # TODO: Implement the test phase forward pass for inverted dropout.   #
#         #######################################################################
#         # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

#         out = x

#         # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
#         #######################################################################
#         #                            END OF YOUR CODE                         #
#         #######################################################################

#     cache = (dropout_param, mask)
#     out = out.astype(x.dtype, copy=False)

#     return out, cache


# ###x
# def dropout_backward(dout, cache):
#     """
#     Perform the backward pass for (inverted) dropout.

#     Inputs:
#     - dout: Upstream derivatives, of any shape
#     - cache: (dropout_param, mask) from dropout_forward.
#     """
#     dropout_param, mask = cache
#     mode = dropout_param['mode']

#     dx = None
#     if mode == 'train':
#         #######################################################################
#         # TODO: Implement training phase backward pass for inverted dropout   #
#         #######################################################################
#         # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

#         dx = dout * mask

#         # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
#         #######################################################################
#         #                          END OF YOUR CODE                           #
#         #######################################################################
#     elif mode == 'test':
#         dx = dout
#     return dx

# ###x
# def conv_forward_naive(x, w, b, conv_param):
#     """
#     A naive implementation of the forward pass for a convolutional layer.

#     The input consists of N data points, each with C channels, height H and
#     width W. We convolve each input with F different filters, where each filter
#     spans all C channels and has height HH and width WW.

#     Input:
#     - x: Input data of shape (N, C, H, W)
#     - w: Filter weights of shape (F, C, HH, WW)
#     - b: Biases, of shape (F,)
#     - conv_param: A dictionary with the following keys:
#       - 'stride': The number of pixels between adjacent receptive fields in the
#         horizontal and vertical directions.
#       - 'pad': The number of pixels that will be used to zero-pad the input. 
        

#     During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
#     along the height and width axes of the input. Be careful not to modfiy the original
#     input x directly.

#     Returns a tuple of:
#     - out: Output data, of shape (N, F, H', W') where H' and W' are given by
#       H' = 1 + (H + 2 * pad - HH) / stride
#       W' = 1 + (W + 2 * pad - WW) / stride
#     - cache: (x, w, b, conv_param)
#     """
#     out = None
#     ###########################################################################
#     # TODO: Implement the convolutional forward pass.                         #
#     # Hint: you can use the function np.pad for padding.                      #
#     ###########################################################################
#     # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
#     st, p = conv_param['stride'], conv_param['pad']
#     N, C, H, W = x.shape
#     F, C, HH, WW = w.shape
#     H_out = 1 + (H+2*p-HH) // st
#     W_out = 1 + (W+2*p-WW) // st
#     X_pad = np.pad(x, [[0,0], [0,0], [p,p], [p,p]], 'constant', constant_values=0)
#     out = np.zeros((N, F, H_out, W_out))
    
#     for i in range(N):
#         for h in range(H_out):
#             for wi in range(W_out):
#                 for f in range(F):
#                     x_small = X_pad[i, :, h*st:h*st+HH, wi*st:wi*st+WW]
#                     w_small = w[f, :,:,:]
#                     out[i,f,h,wi] = (x_small * w_small).sum() + b[f]

#     # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
#     ###########################################################################
#     #                             END OF YOUR CODE                            #
#     ###########################################################################
#     cache = (x, w, b, conv_param)
#     return out, cache

# ###x
# def conv_backward_naive(dout, cache):
#     """
#     A naive implementation of the backward pass for a convolutional layer.

#     Inputs:
#     - dout: Upstream derivatives.
#     - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

#     Returns a tuple of:
#     - dx: Gradient with respect to x
#     - dw: Gradient with respect to w
#     - db: Gradient with respect to b
#     """
#     dx, dw, db = None, None, None
#     ###########################################################################
#     # TODO: Implement the convolutional backward pass.                        #
#     ###########################################################################
#     # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
#     x, w, b, conv_param = cache
#     N, C, H, W = x.shape
#     F, C, HH, WW = w.shape
#     st = conv_param['stride']
#     p = conv_param['pad']
#     x_pad = np.pad(x, ((0, 0), (0, 0), (p, p), (p,p)), 'constant', constant_values=0)
#     H_n = 1 + (H + 2 * p - HH) // st
#     W_n = 1 + (W + 2 * p - WW) // st
#     dx_pad = np.zeros_like(x_pad)
#     dx = np.zeros_like(x)
#     dw = np.zeros_like(w)
#     db = np.zeros_like(b)

#     for n in range(N):
#         for f in range(F):
#             db[f] += dout[n,f].sum()
#             for hi in range(H_n):
#                 for wi in range(W_n):
#                     dw[f] += x_pad[n, :, hi*st:hi*st+HH, wi*st:wi*st+WW] * dout[n,f,hi,wi]
#                     dx_pad[n, :, hi*st:hi*st+HH, wi*st:wi*st+WW] += w[f] * dout[n,f,hi,wi]
                    
#     dx = dx_pad[...,p:p+H, p:p+W]
#     # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
#     ###########################################################################
#     #                             END OF YOUR CODE                            #
#     ###########################################################################
#     return dx, dw, db

# ###x
# def max_pool_forward_naive(x, pool_param):
#     """
#     A naive implementation of the forward pass for a max-pooling layer.

#     Inputs:
#     - x: Input data, of shape (N, C, H, W)
#     - pool_param: dictionary with the following keys:
#       - 'pool_height': The height of each pooling region
#       - 'pool_width': The width of each pooling region
#       - 'stride': The distance between adjacent pooling regions

#     No padding is necessary here. Output size is given by 

#     Returns a tuple of:
#     - out: Output data, of shape (N, C, H', W') where H' and W' are given by
#       H' = 1 + (H - pool_height) / stride
#       W' = 1 + (W - pool_width) / stride
#     - cache: (x, pool_param)
#     """
#     out = None
#     ###########################################################################
#     # TODO: Implement the max-pooling forward pass                            #
#     ###########################################################################
#     # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

#     N, C, H, W = x.shape
#     ph, pw, st = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
#     H_n = int(1 + (H - ph) / st)
#     W_n = int(1 + (W - pw) / st)
#     out = np.zeros((N,C,H_n,W_n))
#     for i in range(N):
#         for hi in range(H_n):
#             for wi in range(W_n):
#                 for l in range(C):
#                     out[i,l,hi,wi] = np.max(x[i,l, hi*st:hi*st+ph, wi*st:wi*st+pw])
#     # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
#     ###########################################################################
#     #                             END OF YOUR CODE                            #
#     ###########################################################################
#     cache = (x, pool_param)
#     return out, cache

# ###x
# def max_pool_backward_naive(dout, cache):
#     """
#     A naive implementation of the backward pass for a max-pooling layer.

#     Inputs:
#     - dout: Upstream derivatives
#     - cache: A tuple of (x, pool_param) as in the forward pass.

#     Returns:
#     - dx: Gradient with respect to x
#     """
#     dx = None
#     ###########################################################################
#     # TODO: Implement the max-pooling backward pass                           #
#     ###########################################################################
#     # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
#     x, pool_param = cache
#     N, C, H, W = x.shape
#     ph, pw, st = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
#     H_n = int(1 + (H - ph) / st)
#     W_n = int(1 + (W - pw) / st)
    
#     dx = np.zeros_like(x)
    
#     for i in range(N):
#         for hi in range(H_n):
#             for wi in range(W_n):
#                 for l in range(C):
#                     index = np.argmax(x[i,l,st*hi:st*hi+ph,st*wi:st*wi+pw])
#                     ind1,ind2 = np.unravel_index(index,(ph, pw))
#                     dx[i,l,hi*st:hi*st+ph, wi*st:wi*st+pw][ind1, ind2] += dout[i,l,hi,wi]

#     # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
#     ###########################################################################
#     #                             END OF YOUR CODE                            #
#     ###########################################################################
#     return dx

    
# ###x
# def softmax_loss(x, y):
#     """
#     Computes the loss and gradient for softmax classification.

#     Inputs:
#     - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
#       class for the ith input.
#     - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
#       0 <= y[i] < C

#     Returns a tuple of:
#     - loss: Scalar giving the loss
#     - dx: Gradient of the loss with respect to x
#     """
#     shifted_logits = x - np.max(x, axis=1, keepdims=True)
#     Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
#     log_probs = shifted_logits - np.log(Z)
#     probs = np.exp(log_probs)
#     N = x.shape[0]
#     loss = -np.sum(log_probs[np.arange(N), y]) / N
#     dx = probs.copy()
#     dx[np.arange(N), y] -= 1
#     dx /= N
#     return loss, dx

from builtins import range
import numpy as np

####x
def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N = x.shape[0]
    D = w.shape[0]
    M = w.shape[1]
    out = x.reshape(N,D).dot(w) + b
    
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


####x
def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    D,M = w.shape
    N = x.shape[0]
    db = np.sum(dout,axis = 0)
    X_ravel = x.reshape((N,D))
    dw = X_ravel.T.dot(dout)
    dX_ravel = dout.dot(w.T)
    dx = dX_ravel.reshape(x.shape)
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


####x
def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    out = x
    out[out<0] = 0
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


####x
def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    dx = np.ones(x.shape)
    dx[x<=0] = 0
    dx = dx*dout
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx





###x
def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        p = dropout_param['p']
        mask = (np.random.rand(*x.shape) < p) / p
        out = x*mask
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        out = x
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


###x
def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        dx = dout*mask
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx

###x
def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
        

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    stride = conv_param['stride']
    pad = conv_param['pad']
    N, C, H, W = x.shape
    Out, C, HH, WW = w.shape
    H_new = int(1 + (H + 2 * pad - HH) / stride)
    W_new = int(1 + (W + 2 * pad - WW) / stride)
    X_padded = np.pad(x,((0,0),(0,0),(pad,pad),(pad,pad)),'constant', constant_values=0)
    out = np.zeros((N,Out,H_new,W_new))
    for i in range(N):
        for j in range(H_new):
            for k in range(W_new):
                for o in range(Out):
                    X_i = X_padded[i]
                    inp_conv = X_i[:,j*stride:j*stride+HH,k*stride:k*stride+WW]
                    out_conv = (inp_conv*w[o,:,:,:]).sum() + b[o]
                    out[i,o,j,k] = out_conv

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache

###x
def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x, w, b, conv_param = cache
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    stride = conv_param['stride']
    pad = conv_param['pad']
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant', constant_values=0)
    H_n = 1 + (H + 2 * pad - HH) // stride
    W_n = 1 + (W + 2 * pad - WW) // stride
    dx_pad = np.zeros_like(x_pad)
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)
    for n in range(N):
        for f in range(F):
            db[f] += dout[n, f].sum()
            for j in range(0, H_n):
                for i in range(0, W_n):
                    dw[f] += x_pad[n,:,j*stride:j*stride+HH,i*stride:i*stride+WW]*dout[n,f,j,i]
                    dx_pad[n,:,j*stride:j*stride+HH,i*stride:i*stride+WW] += w[f]*dout[n, f, j, i]
    dx = dx_pad[:,:,pad:pad+H,pad:pad+W]
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db

###x
def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    H_n = int(1 + (H - pool_height) / stride)
    W_n = int(1 + (W - pool_width) / stride)
    out = np.zeros((N,C,H_n,W_n))
    for i in range(N):
        for j in range(H_n):
            for k in range(W_n):
                for l in range(C):
                    x_max = x[i,l,stride*j:stride*j+pool_height,stride*k:stride*k+pool_width]
                    out[i,l,j,k] = np.amax(x_max)
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache

###x
def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x,pool_param = cache
    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    H_n = int(1 + (H - pool_height) / stride)
    W_n = int(1 + (W - pool_width) / stride)
    dx = np.zeros_like(x)
    for i in range(N):
        for j in range(H_n):
            for k in range(W_n):
                for l in range(C):
                    index = np.argmax(x[i,l,stride*j:stride*j+pool_height,stride*k:stride*k+pool_width])
                    ind1,ind2 = np.unravel_index(index,(pool_height, pool_width))
                    dx[i,l,stride*j:stride*j+pool_height,stride*k:stride*k+pool_width][ind1,ind2] = dout[i,l,j,k]
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx

    
###x
def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
