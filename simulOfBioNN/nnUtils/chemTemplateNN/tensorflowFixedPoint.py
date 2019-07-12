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
def brentq(f, xa, xb, xtol=10**(-12), rtol=4.4408920985006262*10**(-16), iter=100, args=()):
    xpre = xa
    xcur = xb
    xblk = 0.0
    fblk = 0.0
    spre = 0.0
    scur = 0.0

    fpre = f(xpre, args)
    fcur = f(xcur, args)
    if (fpre*fcur > 0):
        return 0.0
    if (fpre == 0):
        return xpre

    if (fcur == 0):
        return xcur

    for i in range(iter):

        if (fpre*fcur < 0):
            xblk = xpre
            fblk = fpre
            spre = xcur - xpre
            scur = xcur - xpre
        if (abs(fblk) < abs(fcur)):
            xpre = xcur
            xcur = xblk
            xblk = xpre

            fpre = fcur
            fcur = fblk
            fblk = fpre

        delta = (xtol + rtol*abs(xcur))/2
        sbis = (xblk - xcur)/2
        if (fcur == 0 or abs(sbis) < delta):
            i = iter #BREAK FAILS HERE!!! ==> strange behavior?
        else:
            if (abs(spre) > delta and abs(fcur) < abs(fpre)):
                if (xpre == xblk):
                # /* interpolate */
                    stry = -fcur*(xcur - xpre)/(fcur - fpre)
                else :
                    # /* extrapolate */
                    dpre = (fpre - fcur)/(xpre - xcur)
                    dblk = (fblk - fcur)/(xblk - xcur)
                    stry = -fcur*(fblk*dblk - fpre*dpre)/(dblk*dpre*(fblk - fpre))

                mymin = tf.minimum(abs(spre), 3*abs(sbis) - delta) #Here would not understand...
                spre=tf.where(tf.less(2*abs(stry)-mymin,0),scur,sbis)
                scur=tf.where(tf.less(2*abs(stry)-mymin,0),stry,sbis)
                #     # /* good short step */
                #if(2*abs(stry) < mymin):
                #     spre = scur
                #     scur = stry
                # else:
                #     # /* bisect */
                #     spre = sbis
                #     scur = sbis
            else:
                # /* bisect */
                spre = sbis
                scur = sbis
            xpre = xcur
            fpre = fcur
            if (abs(scur) > delta):
                xcur += scur
            else:
                if sbis >0:
                    xcur += delta
                else:
                    xcur+= -delta
        fcur = f(xcur, args)
    return xcur



