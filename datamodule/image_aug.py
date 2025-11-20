from typing import Tuple
import numpy as np
import cv2
import random


class ImageAugmentation:
    """Data augmentation for handwritten math formula images."""
    
    def __init__(
        self,
        rotation_range: float = 5.0,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        shear_range: float = 0.1,
        blur_prob: float = 0.3,
        noise_prob: float = 0.3,
        noise_std: float = 0.05,
        brightness_range: Tuple[float, float] = (0.8, 1.2),
        contrast_range: Tuple[float, float] = (0.8, 1.2),
        elastic_prob: float = 0.2,
        elastic_alpha: float = 20.0,
        elastic_sigma: float = 5.0,
    ):
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.shear_range = shear_range
        self.blur_prob = blur_prob
        self.noise_prob = noise_prob
        self.noise_std = noise_std
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.elastic_prob = elastic_prob
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
    
    def apply(self, img: np.ndarray) -> np.ndarray:
        """Apply random augmentations to image."""
        # Geometric transformations
        if random.random() < 0.7:
            img = self._geometric_transform(img)
        
        # Blur
        if random.random() < self.blur_prob:
            img = self._apply_blur(img)
        
        # Gaussian noise
        if random.random() < self.noise_prob:
            img = self._add_noise(img)
        
        # Brightness & contrast
        if random.random() < 0.5:
            img = self._adjust_brightness_contrast(img)
        
        # Elastic deformation
        if random.random() < self.elastic_prob:
            img = self._elastic_transform(img)
        
        return img
    
    def _geometric_transform(self, img: np.ndarray) -> np.ndarray:
        """Apply rotation, scale, and shear."""
        h, w = img.shape[:2]
        center = (w / 2, h / 2)
        
        # Random rotation
        angle = random.uniform(-self.rotation_range, self.rotation_range)
        
        # Random scale
        scale = random.uniform(*self.scale_range)
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, scale)
        
        # Add shear
        shear = random.uniform(-self.shear_range, self.shear_range)
        M[0, 1] += shear
        
        # Apply transformation
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        return img
    
    def _apply_blur(self, img: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur."""
        kernel_size = random.choice([3, 5])
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    
    def _add_noise(self, img: np.ndarray) -> np.ndarray:
        """Add Gaussian noise."""
        noise = np.random.normal(0, self.noise_std * 255, img.shape)
        img = img + noise
        return np.clip(img, 0, 255).astype(np.uint8)
    
    def _adjust_brightness_contrast(self, img: np.ndarray) -> np.ndarray:
        """Adjust brightness and contrast."""
        brightness = random.uniform(*self.brightness_range)
        contrast = random.uniform(*self.contrast_range)
        
        img = img.astype(np.float32)
        img = img * contrast + (brightness - 1) * 128
        return np.clip(img, 0, 255).astype(np.uint8)
    
    def _elastic_transform(self, img: np.ndarray) -> np.ndarray:
        """Apply elastic deformation."""
        h, w = img.shape[:2]
        
        # Generate random displacement fields
        dx = np.random.randn(h, w) * self.elastic_sigma
        dy = np.random.randn(h, w) * self.elastic_sigma
        
        # Smooth the displacement fields
        dx = cv2.GaussianBlur(dx, (0, 0), self.elastic_sigma) * self.elastic_alpha
        dy = cv2.GaussianBlur(dy, (0, 0), self.elastic_sigma) * self.elastic_alpha
        
        # Create mesh grid
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)
        
        # Apply displacement
        img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return img
