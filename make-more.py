import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

words = open('names.txt', 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}


# create the training set of bigrams (x,y)
xs, ys = [], []

for w in words:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    print(ch1, ch2)
    xs.append(ix1)
    ys.append(ix2)
    
xs = torch.tensor(xs)
ys = torch.tensor(ys)

import torch.nn.functional as F
xenc = F.one_hot(xs, num_classes=27).float()

# randomly initialize 27 neurons' weights. each neuron receives 27 inputs
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g)

xenc = F.one_hot(xs, num_classes=27).float() # input to the network: one-hot encoding
logits = xenc @ W # predict log-counts
counts = logits.exp() # counts, equivalent to N
probs = counts / counts.sum(1, keepdims=True) # probabilities for next character
# btw: the last 2 lines here are together called a 'softmax'
print(probs)
nlls = torch.zeros(5)
for i in range(5):
  # i-th bigram:
  x = xs[i].item() # input character index
  y = ys[i].item() # label character index
  print('--------')
  print(f'bigram example {i+1}: {itos[x]}{itos[y]} (indexes {x},{y})')
  print('input to the neural net:', x)
  print('output probabilities from the neural net:', probs[i])
  print('label (actual next character):', y)
  p = probs[i, y]
  print('probability assigned by the net to the the correct character:', p.item())
  logp = torch.log(p)
  print('log likelihood:', logp.item())
  nll = -logp
  print('negative log likelihood:', nll.item())
  nlls[i] = nll

# print('=========')
# print('average negative log likelihood, i.e. loss =', nlls.mean().item())


# create the dataset
xs, ys = [], []
# for w in words[:1][0:1]:
for w in ["emma","emma","emma","emma","emma","emma","emma","emma","emma","emma"]:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    xs.append(ix1)
    ys.append(ix2)
xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()
print('number of examples: ', num)

# initialize the 'network'
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g, requires_grad=True)


# gradient descent
for k in range(1):
  
  # forward pass
  xenc = F.one_hot(xs, num_classes=27).float() # input to the network: one-hot encoding
  
  def get_logits(direct: False):
      if direct:
          import numpy as np
          logits_array = []
          for index, i_c in enumerate(xenc):
            #i_c = [1,0,0....] for ,
            my_array = np.array(i_c.item()) 
            index_value = np.where(my_array == 1)

            logits_array.append(W[index, index_value])
          
          return logits_array


      else:
        return xenc @ W

  logits = xenc @ W

  print(logits)
  print(W[0])

  counts = logits.exp() # counts, equivalent to N
  print(counts)
  probs = counts / counts.sum(1, keepdims=True) # probabilities for next character
  print(probs)

  # print('Probability of e:',probs[[0], [5]])
  # print('Probability of e:',probs[[0], [5]].log())
  # #print('Probability of e:',probs[[0,1,2,3...268000], [5,1,3,15,26,22...]].log())
  # print('Probability of e:',probs[[0], [5]].log().mean() + 0.05*(W**2).mean())



  loss = -probs[torch.arange(num), ys].log().mean() + 0.05*(W**2).mean()
  #probs[torch.arange(num), ys] size is 268000
  #2.56
  #print(loss.item())
  
  # backward pass
  W.grad = None # set to zero the gradient
  loss.backward()

  # update
  W.data += -1 * W.grad
 


  # finally, sample from the 'neural net' model

g = torch.Generator().manual_seed(2147483647)

for i in range(5):
  
  out = []
  ix = 0
  while True:
    
    # ----------
    # BEFORE:
    #p = P[ix]
    # ----------
    # NOW:
    xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()

    def get_logits(direct: False):
      if direct:
          import numpy as np
          logits_array = []
          actual_logits = xenc @ W
          for index, i_c in enumerate(xenc):
            #i_c = [1,0,0....] for ,
            my_array = np.array(i_c.item()) 
            index_value = np.where(my_array == 1)

            logits_array.append(W[index, index_value])
          
         
          return logits_array


      else:
        return xenc @ W

    logits = get_logits(True)
    
    #logits = xenc @ W # predict log-counts
    
    if ix == 0:
      logits[0,2] = 50
    counts = logits.exp() # counts, equivalent to N
    p = counts / counts.sum(1, keepdims=True) # probabilities for next character
    # ----------
    
    ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
    out.append(itos[ix])
    if len(out) > 5:
      break
    if ix == 0:
      break
  print(''.join(out))