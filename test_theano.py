import theano
import theano.tensor as T
from theano.ifelse import ifelse
from utils import *
from utils import *

x = T.vector('dtype=int32')
y = T.vector('dtype=int32')
z = T.vector('dtype=int32')


a = T.vector('dtype=int32')
b = T.vector('dtype=int32')


c = T.matrix('dtype=int32')
d = T.matrix('dtype=int32')


ctx = concatenate([c, d], axis=0)

f1 = theano.function([c,d], ctx, on_unused_input='warn')
#f = theano.function([x,y,z], final, on_unused_input='warn', mode=theano.Mode(linker='vm'))




print f1([[5,6,1,2,3,1,4,0,1],[5,6,1,2,3,1,4,0,1]], [[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9]])


#print f([5,6,1,2,3,1,4,0,1], [1,2,3,4,5,6,7,8,9], [10,11,12,13,14,15,16,17,18])
#print type(f([5,6,1,2,3,1,4,0,1], [1,2,3,4,5,6,7,8,9], [10,11,12,13,14,15,16,17,18]))
