import random 

'''
-   T1 = accuracy threshold
    Pass accuracy acheived from NRO (whatever being used)
-   T2 =  action probability threshold
    Initialized to 0.8 
-   P = action probability vector
    Set to 1/r
-   R = remaining feature set 
    Pass all the features selected by NRO 
-   delta = minimal step size of probability change, defined as Δ=1/αr , where α is the
    resolution parameter.
    Set to 0.00244
-   Q =  LA onoff probability 
    Set to 0.5
-   A = the chosen action set which contains all actions chosen by the LAs in the “ON” state
'''

def choose_on_off (Q):
    x = random.uniform(0,1)
    if x < Q :
        return "OFF"
    else :
        return "ON"

def LA(T1,P, R, T2 = 0.8, delta = 0.00244, Q = 0.5 ):
    alpha, P = 
    A = list()
    t = 0
    while True :
        on_off_state = choose_on_off(Q)
        A = choose_action_set(on_off_state, T2, t)

