# TODO February 21, 2023: Improve this using some configuration files.
#                         For exmple https://protobuf.dev/getting-started/pythontutorial/

import cv2
from pytvision.transforms import transforms as mtrans


def get_transforms_aug(size_input):
    return [
        # ------------------------------------------------------------------
        # Resize
        mtrans.ToResize((size_input + 5, size_input + 5), resize_mode="square"),
        mtrans.RandomCrop((size_input, size_input), limit=2, padding_mode=cv2.BORDER_REPLICATE),
        mtrans.ToGrayscale(),
        # ------------------------------------------------------------------
        # Geometric
        mtrans.RandomScale(factor=0.2, padding_mode=cv2.BORDER_REPLICATE),
        mtrans.ToRandomTransform(
            mtrans.RandomGeometricalTransform(angle=30, translation=0.2, warp=0.02, padding_mode=cv2.BORDER_REPLICATE),
            prob=0.5,
        ),
        mtrans.ToRandomTransform(mtrans.VFlip(), prob=0.5),
        # mtrans.ToRandomTransform( mtrans.HFlip(), prob=0.5 ),
        # ------------------------------------------------------------------
        # Colors
        mtrans.ToRandomTransform(mtrans.RandomBrightness(factor=0.25), prob=0.50),
        mtrans.ToRandomTransform(mtrans.RandomContrast(factor=0.25), prob=0.50),
        mtrans.ToRandomTransform(mtrans.RandomGamma(factor=0.25), prob=0.50),
        mtrans.ToRandomTransform(mtrans.RandomRGBPermutation(), prob=0.50),
        mtrans.ToRandomTransform(mtrans.CLAHE(), prob=0.25),
        mtrans.ToRandomTransform(mtrans.ToGaussianBlur(sigma=0.05), prob=0.25),
        # ------------------------------------------------------------------
        # mtrans.ToEqNormalization([size_input, size_input]),
        mtrans.ToTensor(),
        mtrans.ToNormalization(),
    ]


def get_transforms(size_input):
    return [
        mtrans.ToResize((size_input, size_input), resize_mode="squash"),
        mtrans.ToGrayscale(),
        mtrans.ToTensor(),
        mtrans.ToNormalization(),
    ]
