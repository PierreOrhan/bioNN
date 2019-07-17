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
def brentq(f, xa, xb, xtol=10**(-12), rtol=4.4408920985006262*10**(-16), iter=100, test =[10,10,10,10,10,101,10,10,10,101,1,1,10,10],args=()):

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
        print(i)
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

@tf.function
def g(iter,x):
    t=tf.zeros((10,10))
    for i in tf.range(iter):
        x.assign(i)
        tf.print(x)
        tf.print(i)
        tf.print(t[i])
    return iter

import time
if __name__=="__main__":
    t=time.time()
    # print(brentq(f,tf.constant(-4.),tf.constant(4.)))
    # print(t-time.time())
    # t=time.time()
    # print(brentq(f,tf.constant(-4.),tf.constant(4.)))
    # print(t-time.time())
    # t=time.time()
    # print(brentq(f,tf.constant(-10.),tf.constant(10.)))
    # print(t-time.time())
    # t=time.time()
    x = tf.Variable(tf.constant(0))
    e = g(10,x)
    print(g.__code__)

