import cv2
import numpy as np


img = cv2.imread('/home/peter/tmp/iu.jpg')
org = img.copy()

img1 = img.copy()
img2 = img.copy()
img3 = img.copy()
# print(img.shape)
# h,w= img.shape





def overlaiedfunc(img, center, rad ):

    overlaied = img.copy()
    alpha = 0.2
    for i in range(7):
        mask = np.zeros(img.shape, dtype=np.uint8)
        cv2.circle(mask,center, int(rad*2) - i*10, (255,255,255),-1)
        mask = np.sum(mask,axis=2).astype(np.bool)
        mask = np.logical_not(mask)
        imgk = img.copy()
        imgk[mask] = [255,255,255]
        overlaied = cv2.addWeighted(imgk, alpha, overlaied, 1-alpha, 0)
        
        
        
        
    cv2.imwrite('/home/peter/tmp/iu_circl22.jpg',overlaied)
        
overlaiedfunc(img,(500,400),100)

#     mask1 = np.sum(mask1,axis=2).astype(np.bool)
#     mask1 = np.logical_not(mask1)


#     mask2 = np.sum(mask2,axis=2).astype(np.bool)
#     mask2 = np.logical_not(mask2)


#     mask3 = np.sum(mask3,axis=2).astype(np.bool)
#     mask3 = np.logical_not(mask3)




#     img1[mask1] = [255,255,255]
#     img2[mask2] = [255,255,255]
#     img2[mask3] = [255,255,255]



#     overlayed = cv2.addWeighted(img1, 0.5, org, 0.5, 0)
#     overlayed = cv2.addWeighted(img2, 0.5, overlayed, 0.5, 0)
#     overlayed = cv2.addWeighted(img3, 0.5, overlayed, 0.5, 0)




# cv2.imwrite('/home/peter/tmp/iu_circle.jpg',overlaied)

# a = np.zeros( (5,5,3), dtype=np.uint8)
# cv2.circle(img,())
# a[2,2,:] =3
# print( np.sum(a,axis=2).astype(np.bool))

