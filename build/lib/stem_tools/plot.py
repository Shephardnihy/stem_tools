import numpy as np
from skimage.feature import canny
from scipy.optimize import minimize
import matplotlib


def XYZ2RGB(vx,vy,vz,BrightScale=1.0):
    """
    Writtend by Haoyang Ni

    Convert 3D vector field to a colormap where the hue corresponds to azimuthal angle.V
    Saturation corresponds to vector magnitude in xy-plane, normalized by full 3d vector
    magnitude. Value corresponds to polar angle, where 0 is along -z and 1 is along +z.

    parameters
    ----------
    vx : 2d numpy array
        x component of a vector field
    
    vy : 2d numpy array
        y component of a vector field

    vz : 2d numpy array
        z component of a vector field

    BrightScale : float 
        Multiplier used to increase brightness of image

    returns
    -------
    Cim : (..., 3) numpy array
        RGB image of vector field
    """
    r = np.sqrt(vx**2 + vy**2 + vz**2)
    r_xy = np.sqrt(vx**2 + vy**2)

    hue = (np.pi - np.arctan2(vy, vx)) / (2 * np.pi)
    sat = r_xy/r
    val = (np.arccos(vz / r) / np.pi)
    val = (val - val.min())/(val.max() - val.min())

    hsv = np.stack([hue,sat,val*BrightScale], axis=-1)
    Cim=matplotlib.colors.hsv_to_rgb(hsv)
    return Cim

def XY2RGB(vx, vy,BrightScale=1.0):
    """
    Written by Jordan Hachtel, modifed by Haoyang Ni

    Convert vector field to a colormap where the color corresponds to the angle.

    parameters
    ----------
    vx : 2d numpy array
        x component of a vector field
    
    vy : 2d numpy array
        y component of a vector field

    BrightScale : float 
        Multiplier used to increase brightness of image

    returns
    -------
    Cim : (..., 3) numpy array
        RGB image of vector field
    """

    XY=np.zeros(vx.shape+(3,),dtype=float)
    Eint=np.sqrt(vx**2+vy**2)
    M=np.amax(Eint)
    for i in range(vx.shape[0]):
        for j in range(vy.shape[1]):
            XY[i,j]=(np.angle(vx[i,j]+1j*vy[i,j]))/(2*np.pi)%1,1,Eint[i,j]/M*BrightScale
    Cim=matplotlib.colors.hsv_to_rgb(XY)
    return Cim

def GetLegend(N = 300,r = 0.9,theta = 0):
    """
    Written by Jordan Hachtel, modifed by Haoyang Ni
    Get colorwheel as legend

    
    returns
    -------
    Legend : (..., 3) numpy array
        RGB image of vector field
    """

    xleg=np.linspace(-1,1,N,endpoint=True)
    yleg=np.linspace(-1,1,N,endpoint=True)
    #rxleg,ryleg=xleg*np.cos(theta)-yleg*np.sin(theta),xleg*np.sin(theta)+yleg*np.cos(theta)
    xxl,yyl=np.meshgrid(xleg,yleg)
    mask=xxl**2+yyl**2<r**2
    XL,YL=xxl*mask,yyl*mask
    radtheta = theta*np.pi/180
    RXL,RYL=XL*np.cos(radtheta)+YL*np.sin(radtheta),-XL*np.sin(radtheta)+YL*np.cos(radtheta)
    XY=np.zeros(RXL.shape+(3,),dtype=float)
    I=np.sqrt(RXL**2+RYL**2)
    M=np.amax(I)
    for i in range(RXL.shape[0]):
        for j in range(RXL.shape[1]):
            XY[i,j]=np.angle(RXL[i,j]+1j*RYL[i,j])/(2*np.pi)%1,1,I[i,j]/M
    return matplotlib.colors.hsv_to_rgb(XY)