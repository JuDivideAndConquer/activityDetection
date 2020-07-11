from math import exp

def sigmoid(gamma):
    #print(gamma)
    if gamma < 0:
        return 1 - 1/(1 + exp(gamma))
    else:
        return 1/(1 + exp(-gamma))

while(True):
	n=float(input())
	p=sigmoid(n)
	print(p)
