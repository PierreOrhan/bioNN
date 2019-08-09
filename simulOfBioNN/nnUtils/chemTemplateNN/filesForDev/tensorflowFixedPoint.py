"""
    In this module we simply adapt the brentq algorithm from scipy in tensorflow 2.0.

    adapted from a code by Charles Harris charles.harris@sdl.usu.edu  in c.
"""
# /*
# At the top of the loop the situation is the following:
# 1. the root is bracketed between xa and xb
# 2. xa is the most recent estimate
# 3. xp is the previous estimate
# 4. |fp| < |fb|
# The order of xa and xp doesn't matter, but assume xp < xb. Then xa lies to
# the right of xp and the assumption is that xa is increasing towards the root.
# In this situation we will attempt quadratic extrapolation as long as the
# condition
# *  |fa| < |fp| < |fb|
# is satisfied. That is, the function value is decreasing as we go along.
# Note the 4 above implies that the right inequlity already holds.
# The first check is that xa is still to the left of the root. If not, xb is
# replaced by xp and the interval reverses, with xb < xa. In this situation
# we will try linear interpolation. That this has happened is signaled by the
# equality xb == xp;
# The second check is that |fa| < |fb|. If this is not the case, we swap
# xa and xb and resort to bisection.
# */
import tensorflow as tf

@tf.function
def brentq(f, xa, xb, iter,xtol=10**(-12), rtol=4.4408920985006262*10**(-16), test =[10,10,10,10,10,101,10,10,10,101,1,1,10,10],args=()):

    # test2 = []
    # for t in tf.range(len(test)):
    #     print(t)
    #     test2 += [test[t]]
    xpre = xa
    xcur = xb
    xblk = tf.constant(0.)
    fblk = tf.constant(0.)
    spre = tf.constant(0.)
    scur = tf.constant(0.)
    fpre = f(xpre, args)
    fcur = f(xcur, args)
    if tf.math.greater(fpre*fcur,0):
        return 0.0
    if tf.equal(fpre,0):
        return xpre
    if tf.equal(fcur,0):
        return xcur

    for i in tf.range(iter):
        if tf.less(fpre*fcur,0):
            xblk = xpre
            fblk = fpre
            spre = xcur - xpre
            scur = xcur - xpre
        if tf.less(tf.abs(fblk),tf.abs(fcur)):
            xpre = xcur
            xcur = xblk
            xblk = xpre

            fpre = fcur
            fcur = fblk
            fblk = fpre

        delta = (xtol + rtol*tf.abs(xcur))/2
        sbis = (xblk - xcur)/2
        if tf.equal(fcur,0) or tf.less(tf.abs(sbis),delta):
            i = iter #BREAK FAILS HERE!!! ==> strange behavior?
        else:
            if tf.greater(tf.abs(spre),delta) and tf.less(tf.abs(fcur),tf.abs(fpre)):
                if tf.equal(xpre,xblk):
                # /* interpolate */
                    stry = -fcur*(xcur - xpre)/(fcur - fpre)
                else :
                    # /* extrapolate */
                    dpre = (fpre - fcur)/(xpre - xcur)
                    dblk = (fblk - fcur)/(xblk - xcur)
                    stry = -fcur*(fblk*dblk - fpre*dpre)/(dblk*dpre*(fblk - fpre))

                mymin = tf.minimum(tf.abs(spre), 3*tf.abs(sbis) - delta) #Here would not understand...
                spre=tf.where(tf.less(2*tf.abs(stry)-mymin,0),scur,sbis)
                scur=tf.where(tf.less(2*tf.abs(stry)-mymin,0),stry,sbis)
            else:
                # /* bisect */
                spre = sbis
                scur = sbis
            xpre = xcur
            fpre = fcur
            if tf.greater(tf.abs(scur),delta):
                xcur += scur
            else:
                if tf.greater(sbis,0):
                    xcur += delta
                else:
                    xcur += -delta
        fcur = f(xcur, args)
    return xcur

@tf.function
def f(x,args):
    return (x-1)**3
#
# @tf.function
# def g(iter,x):
#     t=tf.zeros((10,10))
#     for i in tf.range(iter):
#         x.assign(i)
#         tf.print(x)
#         tf.print(i)
#         tf.print(t[i])
#     return iter
# @tf.function
# def g(m):
#     for i in tf.range(4):
#         z =tf.zeros((i,i))
#         tf.print(z)
#         m = tf.sequence_mask([i]*4,4)
#         tf.print(tf.boolean_mask(tf.zeros((4,4)),m))
#



class VariableRaggedTensor(tf.Module):
    def __init__(self,raggedTensor):
        self.var_rowsplits = tf.Variable(raggedTensor.row_splits,trainable=False)
        self.multiDim = False
        if type(raggedTensor.values)==tf.RaggedTensor:
            self.var_values = VariableRaggedTensor(raggedTensor.values)
            self.multiDim = True
        else:
            self.var_values = tf.Variable(raggedTensor.values,trainable=False)
        self.shape0 = tf.Variable(raggedTensor.shape[0],trainable=False)
    @tf.function
    def getRagged(self):
        if self.multiDim:
            return tf.RaggedTensor.from_row_splits(row_splits=self.var_rowsplits,values=self.var_values.getRagged())
        return tf.RaggedTensor.from_row_splits(row_splits=self.var_rowsplits,values=self.var_values)

@tf.function
def f(x):
    tf.print(tf.shape(x)[0])
    return tf.shape(x)

@tf.function
def g(x):
    e =tf.TensorArray(dtype=tf.float32,size=1)
    for idx in tf.range(2):
        if tf.equal(idx,0):
            e=e.write(0,tf.shape(x)[0])
        else:
            tf.print(e.read(0))
    return x

@tf.function
def tryBreak():
    for idx in tf.range(100):
        tf.print(idx)
        if(tf.equal(idx,10)):
            break

@tf.function
def raggedStress(raggedTensor):
    zero = tf.zeros((10,10),dtype=tf.float32)
    v = tf.fill([1],1.)
    for idx in tf.range(10):
        v += tf.keras.backend.sum(raggedTensor[idx].to_tensor()*zero)
    return v

def normalStress(list):
    zero = np.zeros((10,10))
    v = 1
    for idx in range(10):
        v += np.sum(list[idx]*zero)
    return v


import time
import numpy as np
if __name__=="__main__":
    # t=time.time()
    # z =tf.TensorShape((100,None))
    # layerlist = np.zeros(100)
    # m=[(10,10)]*100
    # z = tf.stack([tf.RaggedTensor.from_tensor(tf.zeros(m[idx],dtype=tf.float32)) for idx in range(layerlist.shape[0])])
    # print(z[0].shape)
    # vz = VariableRaggedTensor(z)
    # print(vz.getRagged()[0].shape)
    # print(tf.constant(1.))
    # print(g().shape)
    #e =tf.TensorArray(dtype=tf.int32,size=1,clear_after_read=False)
    #e =tf.Variable(initial_value=0)
    # e = tf.zeros(10)
    g(tf.zeros(10))

    # for _ in range(10):
    #     myRaggedTensor = tf.stack([tf.RaggedTensor.from_tensor(tf.ones((10,10))) for _ in tf.range(10)])
    #     myTensor = np.stack([np.ones((10,10))] for _ in range(10))
    #     t0 = time.time()
    #     a = raggedStress(myRaggedTensor)
    #     print("computed tf raggged in "+str(time.time()-t0))
    #     t0 = time.time()
    #     b = normalStress(myTensor)
    #     print("computed numpy raggged in "+str(time.time()-t0))


    #tryBreak()
    #print(tf.map_fn(f,tf.zeros((5,10,1)),dtype=tf.int32))

    #print(brentq(f,tf.constant(-4.),tf.constant(4.),iter=z.shape[0]))
    # print(t-time.time())
    # t=time.time()
    # print(brentq(f,tf.constant(-4.),tf.constant(4.)))
    # print(t-time.time())
    # t=time.time()
    # print(brentq(f,tf.constant(-10.),tf.constant(10.)))
    # print(t-time.time())
    # t=time.time()
    # x = tf.Variable(tf.constant(0))
    # e = g(10,x)
    # print(g.__code__)
    # mask = tf.sequence_mask([2]*10,10)
    # g(mask)

