from libertem.udf import UDF
from skimage.transform import downscale_local_mean
from skimage.feature import canny
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift, rotate
import numpy as np
from .utils import *

class PreprocessFRMS6(UDF):
    def __init__(self, gainmap, hardware_bin = 1,software_bin = 1, rotation = 0, shifts = None,  *args, **kwargs):
        """
        Preprocess 4D-STEM dataset, optional binning, rotation and descan correction can be applied.

        Parameters
        ----------
        gainmap : ndarray
            Gain reference of PNCCD. Gainmap will be further binned with harware binning.

        harware_bin : int
            PNCCD acquisition binning, 1 (264*264), 2 (132*132) or 4 (66*66)
        
        software_bin : int
            Further binning using local mean descale after harware bin
        
        rotation : float
            Rotation angle between scanning and detector
        
        shifts : aux_data of ndarray
            shifts needs to be applied to 4D-STEM dataset to compensate for imperfect descanning, should be a (sx, sy, 2) array
        """
        super().__init__(*args, gainmap=gainmap, hardware_bin = hardware_bin, software_bin = software_bin, rotation = rotation, shifts = shifts, **kwargs)

    def get_result_buffers(self):
        sx, sy = self.meta.dataset_shape[2], self.meta.dataset_shape[3]
        hardware_bin = self.params.hardware_bin
        software_bin = self.params.software_bin

        return {
            "pattern": self.buffer(kind = 'nav', dtype = np.int16, extra_shape = (sx//hardware_bin//software_bin,sy//hardware_bin//software_bin)),
        }
    
    def process_frame(self, frame):
        
        hardware_bin = self.params.hardware_bin
        software_bin = self.params.software_bin
        rotation = self.params.rotation
        gainmap = downscale_local_mean(self.params.gainmap, (1, hardware_bin))

        frame_gain_corrected = downscale_local_mean(frame, hardware_bin)*gainmap
        #print(shifts)
        if self.params.shifts is None:
            frame_rotated = rotate(frame_gain_corrected, rotation, reshape = False)
            frame_binned = downscale_local_mean(frame_rotated, software_bin)
        else:
            shifts = self.params.shifts[:]
            frame_shifted = shift(frame_gain_corrected , shifts)
            frame_rotated = rotate(frame_shifted, rotation, reshape = False)
            frame_binned = downscale_local_mean(frame_rotated, software_bin)
            
        frame_final = frame_binned.astype('int16')
        self.results.pattern[:] = frame_final[:]


class FindDiskCenterEllipseFitting(UDF):
    def __init__(self, sigma = 1, low_threshold= 0.3, high_threshold = 1, *args, **kwargs):
        """
        Find 4D-STEM Disk Center using Ellipse Fitting. Ellipse Fitting is useful when diffraction disk has a well-defined edge and only have one disk. 
        For example, atomic resolution 4D-STEM dataset on a thin sample or 4D-STEM dataset acquired on vacuum region for descan calibration.

        Parameters
        ----------
        threshold : float
            Intensity threshold to binarized the image for Canny edge detection
        """
        super().__init__(*args, sigma = sigma, low_threshold= low_threshold, high_threshold = high_threshold, **kwargs)

    def get_result_buffers(self):
        return {
            "center": self.buffer(kind = 'nav', dtype = np.float64, extra_shape = (2,)),
        }


    def process_frame(self, frame):
        low_threshold = self.params.low_threshold
        high_threshold = self.params.high_threshold
        sigma = self.params.sigma

        edge = canny(frame, sigma = sigma, low_threshold= low_threshold,high_threshold = high_threshold,use_quantiles=True)
        pts = np.argwhere(edge)
        x, y = pts[:,0],pts[:,1]
        coeffs = FitEllipse(x,y)
        x0, y0, ap, bp, e, phi = Cartesian2Polar(coeffs)
        self.results.center[:] = (x0, y0)


class VirtualFieldImaging(UDF):
    def __init__(self, cx, cy, rin, rout, *args, **kwargs):
        """
        Calculate virtual field

        parameters
        ----------
        cx : float
            center along x (column)
        
        cy : float
            center along y (row)

        rin : float
            inner cutoff
        
        rout : float
            outer cutoff
        """
        super().__init__(*args, cx = cx, cy = cy, rin = rin, rout = rout,**kwargs)

    def get_result_buffers(self):
        return {
            "virtual_image": self.buffer(kind = 'nav', dtype = np.float64),
        }
    
    def process_frame(self, frame):
        cx, cy, rin, rout = self.params.cx, self.params.cy, self.params.rin, self.params.rout
        sx, sy = self.meta.dataset_shape[2], self.meta.dataset_shape[3]
        sxx, syy = np.mgrid[0:sx, 0:sy]            
        mask = ((sxx - cx)**2 + (syy - cy)**2 < rout**2)&((sxx - cx)**2 + (syy - cy)**2 > rin**2)

        self.results.virtual_image[:] = (frame*mask).sum()


class FindDiskShiftCrossCorrelation(UDF):
    def __init__(self, reference_img, upsample_factor = 16, *args, **kwargs):
        """
        Find 4D-STEM Disk Deflection using Cross-Correlation.
        parameters
        ----------
        reference_img : ndarray
            Reference image for cross-correlation
        upsample_factor : int
            subpixel accuracy
        """
        super().__init__(*args, reference_img = reference_img, upsample_factor = upsample_factor)
    
    def get_result_buffers(self):
        return {
            "shift": self.buffer(kind = 'nav', dtype = np.float64, extra_shape = (2,))
        }
    
    def process_frame(self, frame):
        reference_img = self.params.reference_img
        upsample_factor = self.params.upsample_factor
        
        s, _, _ = phase_cross_correlation(reference_image = reference_img, 
                                          moving_image = frame,
                                          upsample_factor=upsample_factor)
        self.results.shift[:] = s