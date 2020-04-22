
import cv2
from torchvision import transforms
from pytvision.transforms import transforms as mtrans


# mnist
normalize_mnist = mtrans.ToMeanNormalization(
   mean=[0.1307, 0.1307, 00.1307], std=[0.3081, 0.3081, 0.3081],
   )

# cifar10
normalize_cifar10 = mtrans.ToMeanNormalization(
    mean = (0.4914, 0.4822, 0.4465),
    std  = (0.2023, 0.1994, 0.2010),
    )

# cifar100
normalize_cifar100 = mtrans.ToMeanNormalization(
   mean = [x / 255 for x in [129.3, 124.1, 112.4]],
   std = [x / 255 for x in [68.2, 65.4, 70.4]],
   )

# svhn
normalize_svhn = mtrans.ToMeanNormalization(
   mean = [x / 255 for x in [127.5, 127.5, 127.5]],
   std = [x / 255 for x in [127.5, 127.5, 127.5]],
   )

#TODO: Parametrize normalization
normalize = normalize_mnist #mtrans.ToNormalization()


def get_transforms_aug( size_input ):
    return transforms.Compose([

        #------------------------------------------------------------------
        #Resize
        mtrans.ToResize( (size_input+5, size_input+5), resize_mode='square' ) ,
        mtrans.RandomCrop( (size_input, size_input), limit=2, padding_mode=cv2.BORDER_REPLICATE  ) ,
        mtrans.ToGrayscale(),

        #------------------------------------------------------------------
        #Geometric

        mtrans.RandomScale(factor=0.2, padding_mode=cv2.BORDER_REPLICATE ),
        mtrans.ToRandomTransform( mtrans.RandomGeometricalTransform( angle=30, translation=0.2, warp=0.02, padding_mode=cv2.BORDER_REPLICATE ), prob=0.5 ),
        mtrans.ToRandomTransform( mtrans.VFlip(), prob=0.5 ),
        #mtrans.ToRandomTransform( mtrans.HFlip(), prob=0.5 ),

        #------------------------------------------------------------------
        #Colors
        mtrans.ToRandomTransform( mtrans.RandomBrightness( factor=0.25 ), prob=0.50 ),
        mtrans.ToRandomTransform( mtrans.RandomContrast( factor=0.25 ), prob=0.50 ),
        mtrans.ToRandomTransform( mtrans.RandomGamma( factor=0.25 ), prob=0.50 ),
        mtrans.ToRandomTransform( mtrans.RandomRGBPermutation(), prob=0.50 ),
        mtrans.ToRandomTransform( mtrans.CLAHE(), prob=0.25 ),
        mtrans.ToRandomTransform(mtrans.ToGaussianBlur( sigma=0.05 ), prob=0.25 ),

        #------------------------------------------------------------------
        mtrans.ToTensor(),
        normalize,

        ])


def get_transforms_det(size_input):
    return transforms.Compose([
        #mtrans.ToResize( (38, 38), resize_mode='squash', padding_mode=cv2.BORDER_REPLICATE ),
        #mtrans.ToPad( 5 , 5, padding_mode=cv2.BORDER_REPLICATE ) ,
        #mtrans.ToResize( (size_input+20, size_input+20), resize_mode='squash', padding_mode=cv2.BORDER_REPLICATE ),
        #mtrans.CenterCrop( (size_input, size_input), padding_mode=cv2.BORDER_REPLICATE  ) ,
        mtrans.ToResize( (size_input, size_input), resize_mode='squash' ) ,
        #mtrans.ToResize( (size_input, size_input), resize_mode='square', padding_mode=cv2.BORDER_REPLICATE ) ,
        mtrans.ToGrayscale(),
        mtrans.ToTensor(),
        normalize,
        ])



def get_transforms_fer_aug( size_input ):
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
        mtrans.RandomCrop( (size_input, size_input), limit=2, padding_mode=cv2.BORDER_REFLECT  ),

        #------------------------------------------------------------------
        mtrans.ToGrayscale(),
        mtrans.ToTensor(),
        normalize,

        ])



def get_transforms_fer_det(size_input):
    return transforms.Compose([
        mtrans.ToResize( (size_input, size_input), resize_mode='square', padding_mode=cv2.BORDER_REFLECT ),
        mtrans.ToGrayscale(),
        mtrans.ToTensor(),
        normalize,
        ])
