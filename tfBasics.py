import tensorflow as tf


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------

#**
    # Multiplication of two contstan number
# *#

x1 = tf.constant(5)
x2 = tf.constant(6)

result = tf.multiply(x1,x2)


# Result will come as a tensor
print(result)
# outuput with out session:-
# Tensor("Mul:0", shape=(), dtype=int32)

# Through session we can see the output value without the tensor
sess = tf.Session()
print(sess.run(result))
sess.close()
# outuput with session:-
# 30

# or it wii close the fun automatically
with tf.Session() as sess:
    output = sess.run(result)
    print(output)

print(output)

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------


#**
    # Multiplication of two matrix number
# *#
#
#
# x1 = tf.constant([5])
# x2 = tf.constant([6])
#
#
# result = tf.matmul(x1,x2)
#
# print(result)
#
#
# sess = tf.Session()
# print(sess.run(result))


