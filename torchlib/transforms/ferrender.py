

import numpy as np
import skimage.color as skcolor
import scipy.ndimage as ndi
import cv2
import random as rn

from pytvision.transforms import functional as F


def pad( image, h_pad, w_pad, padding=cv2.BORDER_CONSTANT ):
    image = F.pad(image, h_pad, w_pad, padding_mode)
    return image
        
def scale(image, mask, factor=0.2, padding=cv2.BORDER_CONSTANT ):
    factor =  1.0 + factor*rn.uniform(-1.0, 1.0)    
    image = F.scale( image, factor, cv2.INTER_LINEAR, padding )
    mask  = F.scale( mask, factor, cv2.INTER_NEAREST, padding )    
    return image, mask

def crop(image, cropsize, limit=10, padding=cv2.BORDER_CONSTANT):     
    h, w = image.shape[:2]
    newW, newH = cropsize
    x = rn.randint( -limit, (w - newW) + limit )
    y = rn.randint( -limit, (h - newH) + limit )
    box = [ x, y, cropsize[0], cropsize[1] ]    
    image = F.imcrop( image, box, padding )
    return image

def transform(image, mask, angle=360, translation=0.2, warp=0.0, padding=cv2.BORDER_CONSTANT ):
    imsize = image.shape[:2]
    mat_r, mat_t, mat_w = F.get_geometric_random_transform( imsize, angle, translation, warp )
    image = F.applay_geometrical_transform( image, mat_r, mat_t, mat_w, cv2.INTER_LINEAR , padding )
    mask  = F.applay_geometrical_transform( mask, mat_r, mat_t, mat_w, cv2.INTER_NEAREST , padding )
    return image, mask
    
def norm(image, mask=None):
    
    image = image.astype(np.float)
    for i in range(3):
        image_norm = image[:,:,i]
        minn = np.min( image_norm[mask==1] if mask is not None else image_norm )
        maxn = np.max( image_norm[mask==1] if mask is not None else image_norm )        
        image[:,:,i] = (image_norm-minn)/maxn
    image  =  (image*255.0).astype(np.uint8)   
    return image
        
class Generator(object):
    
    def __init__(self, iluminate=True, angle=45, translation=0.3, warp=0.1, factor=0.2 ):
        self.iluminate=iluminate
        self.angle=angle
        self.translation=translation
        self.warp=warp
        self.factor=factor
        


    def mixture(self, img, mask, back, iluminate=True, angle=45, translation=0.3, warp=0.1, factor=0.2 ):
        '''mixture image with background '''        

        image = img.copy()
        mask  = mask.copy()
        mask  = (mask[:,:,0] == 0).astype(np.uint8)

        #normalizate
        #image  =  image.astype(np.float32) - np.min( image )
        #image /=  np.max(image )
        #image  =  (image*255.0).astype(np.uint8)  
        
        image = norm(image, mask)
        back  = norm(back)   
        
        #scale 
        image, mask = scale( image, mask, factor=factor )
        image, mask = transform( image, mask, angle=angle, translation=translation, warp=warp )        
        image_ilu = image.copy()
        
        #normalize illumination change
        if iluminate:            
            face_lab = skcolor.rgb2lab( image )
            back_lab = skcolor.rgb2lab( back )                      
            l_f = face_lab[:,:,0].mean()
            l_b = back_lab[:,:,0].mean()
            w_ligth = l_b/(l_f + np.finfo(np.float).eps)            
            w_ligth = np.clip( w_ligth, 0.5, 1.5 )
            face_lab[:,:,0] = np.clip( face_lab[:,:,0]*w_ligth , 10, 90 )
            image_ilu = skcolor.lab2rgb(face_lab)*255 
            
            #image_lab = skcolor.rgb2lab( image )
            #image_lab[:,:,0] = image_lab[:,:,0]*(rn.random()+0.5);
            #image_lab[:,:,0] = np.clip( image_lab[:,:,0], 10, 100 )
            #image            = skcolor.lab2rgb(image_lab)*255  
            
        
        se = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7)) #MORPH_CLOSE
        mask = cv2.morphologyEx(mask*1.0, cv2.MORPH_CLOSE, se)
        mask = cv2.erode(mask*1.0, se, iterations=1)==1          
        mask = ndi.morphology.binary_fill_holes( mask*1.0 , structure=np.ones((7,7)) ) == 1
        mask = np.stack( (mask,mask,mask), axis=2 )

        image_org = back*(1-mask) + (mask)*image   
        image_ilu = back*(1-mask) + (mask)*image_ilu
        
        return image_org, image_ilu, mask
    

    def generate(self, image, back, pad = 10 ):
        '''generate image
        '''
        
        image = cv2.resize(image, (256,256) ) 
        
        im_h,im_w = image.shape[:2]
        bk_h,bk_w = back.shape[:2]
        
        #pad
        image_pad = np.zeros( ( im_h+2*pad, im_w+2*pad, 3  ) )
        image_pad[ pad:-pad, pad:-pad, :  ] =  image
        image = image_pad
        im_h,im_w = im_h+2*pad, im_w+2*pad
        
        mask = ( image < 1.0 ).astype( np.uint8 )
        
        dz = 50*rn.random()
        dx = int( rn.random() * ( (bk_w + dz) - im_w ) )
        dy = int( rn.random() * ( (bk_h + dz) - im_h ) )
        back = back[ dy:(dy+im_h), dx:(dx+im_w), : ]
        back = cv2.resize(back, (im_w,im_h) ) 
        
        image_org, image_ilu, mask = self.mixture( image, mask, back, self.iluminate, self.angle, self.translation, self.warp, self.factor  )
        mask = mask.astype(int)
        
        return image_org, image_ilu, mask
        