
import cv2
from torchvision import transforms
from pytvision.transforms import transforms as mtrans


# transformations 
#normalize = mtrans.ToMeanNormalization(
#    #mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
#    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5],
#    )

# cifar10
normalize = mtrans.ToMeanNormalization(
    mean = (0.4914, 0.4822, 0.4465), #[x / 255 for x in [125.3, 123.0, 113.9]],
    std  = (0.2023, 0.1994, 0.2010), #[x / 255 for x in [63.0, 62.1, 66.7]],
    )

# cifar100
#normalize = mtrans.ToMeanNormalization(
#    mean = [x / 255 for x in [129.3, 124.1, 112.4]],
#    std = [x / 255 for x in [68.2, 65.4, 70.4]],
#    )

# svhn
#normalize = mtrans.ToMeanNormalization(
#    mean = [x / 255 for x in [127.5, 127.5, 127.5]],
#    std = [x / 255 for x in [127.5, 127.5, 127.5]],
#    )


# normalize = mtrans.ToNormalization()

def get_transforms_aug( size_input ):        
    return transforms.Compose([
  
        #------------------------------------------------------------------
        #Resize input
         mtrans.ToResize( (48, 48), resize_mode='square', padding_mode=cv2.BORDER_REFLECT ),
                
        #------------------------------------------------------------------
        #Geometric         
        mtrans.RandomScale(factor=0.2, padding_mode=cv2.BORDER_REPLICATE ), 
        mtrans.ToRandomTransform( mtrans.RandomGeometricalTransform( angle=30, translation=0.2, warp=0.02, padding_mode=cv2.BORDER_REFLECT ), prob=0.5 ),
        mtrans.ToRandomTransform( mtrans.VFlip(), prob=0.5 ),
        #mtrans.ToRandomTransform( mtrans.HFlip(), prob=0.5 ),
        
        #------------------------------------------------------------------
        #Colors 
        
        mtrans.ToRandomTransform( mtrans.RandomBrightness( factor=0.25 ), prob=0.50 ),
        mtrans.ToRandomTransform( mtrans.RandomContrast( factor=0.25 ), prob=0.50 ),
        mtrans.ToRandomTransform( mtrans.RandomGamma( factor=0.25 ), prob=0.50 ),
        mtrans.ToRandomTransform( mtrans.RandomRGBPermutation(), prob=0.50 ),
        mtrans.ToRandomTransform( mtrans.CLAHE(), prob=0.25 ),
        mtrans.ToRandomTransform( mtrans.ToGaussianBlur( sigma=0.05 ), prob=0.25 ),

        
        #------------------------------------------------------------------
        #Resize
        mtrans.ToResize( (size_input+5, size_input+5), resize_mode='square', padding_mode=cv2.BORDER_REFLECT ),
        mtrans.RandomCrop( (size_input, size_input), limit=2, padding_mode=cv2.BORDER_REPLICATE  ), 
        
        
        #------------------------------------------------------------------
        mtrans.ToGrayscale(),
        mtrans.ToTensor(),
        normalize,
        
        ])    
    


def get_transforms_det(size_input):    
    return transforms.Compose([       
        mtrans.ToResize( (size_input, size_input), resize_mode='squash', padding_mode=cv2.BORDER_REFLECT ),
        mtrans.ToGrayscale(),
        mtrans.ToTensor(),
        normalize,
        ])



# tta

def get_transforms_hflip(size_input):    
    transforms_hflip = transforms.Compose([
        #mtrans.ToResize( (size_input, size_input), resize_mode='crop' ) ,
        mtrans.ToResize( (size_input, size_input), resize_mode='square', padding_mode=cv2.BORDER_REFLECT ) ,
        mtrans.HFlip(),
        mtrans.ToTensor(),
        normalize,
        ])
    return transforms_hflip

def get_transforms_gray(size_input):    
    transforms_gray = transforms.Compose([
        #mtrans.ToResize( (size_input, size_input), resize_mode='crop' ) ,
        mtrans.ToResize( (size_input, size_input), resize_mode='square', padding_mode=cv2.BORDER_REFLECT ) ,
        mtrans.ToGrayscale(),
        mtrans.ToTensor(),
        normalize,
        ])
    return transforms_gray


def get_transforms_aug2( size_input ):    
    transforms_aug = transforms.Compose([
        
        mtrans.ToResize( (size_input, size_input), resize_mode='square', padding_mode=cv2.BORDER_REFLECT ) ,
        #mtrans.ToResize( (size_input+20, size_input+20), resize_mode='asp' ) ,
        #mtrans.RandomCrop( (size_input, size_input), limit=0, padding_mode=cv2.BORDER_REFLECT_101  ) , 

        mtrans.RandomScale(factor=0.2, padding_mode=cv2.BORDER_REPLICATE ), 
        mtrans.RandomGeometricalTransform( angle=45, translation=0.2, warp=0.02, padding_mode=cv2.BORDER_REFLECT),
        mtrans.ToRandomTransform( mtrans.HFlip(), prob=0.5 ),
        #------------------------------------------------------------------
        #mtrans.RandomRGBPermutation(),
        #mtrans.ToRandomChoiceTransform( [
        #    mtrans.RandomBrightness( factor=0.15 ), 
        #    mtrans.RandomContrast( factor=0.15 ),
        #    #mtrans.RandomSaturation( factor=0.15 ),
        #    mtrans.RandomHueSaturation( hue_shift_limit=(-5, 5), sat_shift_limit=(-11, 11), val_shift_limit=(-11, 11) ),
        #    mtrans.RandomGamma( factor=0.30  ),            
        #    mtrans.ToRandomTransform(mtrans.ToGrayscale(), prob=0.15 ),
        #    ]),    
        #mtrans.ToRandomTransform(mtrans.ToGaussianBlur( sigma=0.0001), prob=0.15 ),    

        #------------------------------------------------------------------
        mtrans.ToTensor(),
        normalize,
        ])    
    return transforms_aug

