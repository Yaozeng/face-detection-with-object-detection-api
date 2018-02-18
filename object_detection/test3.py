##########################################################
"""
import cv2
clicked=False
def onMounse(event,x,y,param):
    global clicked
    if event==cv2.EVENT_LBUTTONUP:
        clicked=True
camerCapture=cv2.VideoCapture(0)
cv2.namedWindow('mywindows')
cv2.setMouseCallback('mywindows',onMounse)

success,frame=camerCapture.read()
while success and cv2.waitKey(1)==-1 and not clicked:
    cv2.imshow('mywindows',frame)
    success,frame=camerCapture.read()
cv2.destroyWindow('mywindows')
camerCapture.release()
"""
#########################################################
"""
import tensorflow as tf
a=tf.random_normal([1,28,28,3])
#b=tf.layers.max_pooling2d(a,[3,3],2,padding="valid")
#b=tf.nn.max_pool(a,[1,3,3,1],[1,2,2,1],padding="VALID")
b=tf.layers.conv2d(a,16,[3,3],2,padding="same")
print(b.shape)
#c=tf.layers.max_pooling2d(a,[3,3],2,padding="same")
#c=tf.nn.max_pool(a,[1,3,3,1],[1,1,1,1],padding="SAME")
c=tf.layers.conv2d(a,16,[3,3],2,padding="valid")
print(c.shape)
"""
print(2**3)