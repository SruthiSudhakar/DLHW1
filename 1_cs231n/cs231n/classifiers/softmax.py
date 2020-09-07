import numpy as np
from random import shuffle
import pdb
def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
    
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  epsilon = 1e-9
  predictions = np.dot(W,X)
  predictions -= np.amax(predictions,axis=0)
  p = np.exp(predictions)
  sums = np.sum(p, axis=0)
  all_probabilities = p/(sums)
  correct_probabilities = all_probabilities[y,np.arange(len(y))]
    
  regularizer =  0.5*reg*np.sum(W*W) #0.5 to account for gradient overflow/underflow
  loss = -np.sum(np.log(correct_probabilities + epsilon))/X.shape[1] + regularizer
  
  dlds = all_probabilities
  dlds[y,range(len(y))] -= 1
  dlds/=len(y)
  dW = np.dot(dlds,X.T) + reg*W
  '''print('predictions: ',np.argwhere(np.isnan(predictions)))
  print('p:',np.argwhere(np.isnan(p)))
  print('sums: ',np.argwhere(sums==0))
  print('dlds: ',np.argwhere(np.isnan(dlds)))
  print('correct_probabilities',correct_probabilities)
  print('predictions',predictions)
  print('p',p)
  print('sums',sums)
  print('all_probabilities',all_probabilities)
  print('correct_probabilities',correct_probabilities)
  print('loss', loss)

  print('dlds-1: ',dlds, np.argwhere(np.isnan(dlds)))
  print('dlds/n: ',dlds,np.argwhere(np.isnan(dlds)))
  print('reg*W: ',reg*W, np.argwhere(np.isnan(reg*W)))
  print('XT: ',X.T)
  print('dlds dot XT: ',np.dot(dlds,X.T), np.argwhere(np.isnan(np.dot(dlds,X.T))))
  print('dw: ',dW, np.argwhere(np.isnan(np.dot(dlds,X.T))))'''
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
