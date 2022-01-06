#1/5/2022   Michael Ofengenden & Roger Yu & A LOT A LOT of help from Zack Gainsforth
import sys, os
import hyperspy.api as hs
import numpy as np
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import tifffile as tif
import skimage
from skimage.transform import resize
from skimage.io import imsave
from numba import jit
import argparse
import matplotlib.pyplot as plt

def makeSpectrum(EDS, mask, cubemask = False):
    
    #creating copy
    EDS_copy = EDS.data.copy()
    #multiplying the stack with mask using numba
    @jit
    def domult(EDS_copy, mask):
        for i in range(len(EDS_copy)):
            for j in range(len(EDS_copy[0])):
                EDS_copy[i][j] = EDS_copy[i][j] * mask[i][j]
        return EDS_copy

    #check if mask is a multiple of the dimensions of the stack
    if mask.shape[0] % EDS_copy.shape[0] == 0:
        BinningFactor = EDS_copy.shape[0] / mask.shape[0]
        if EDS_copy.shape[1] == mask.shape[1] * BinningFactor:
            if BinningFactor != 1:
                print('Resizing Mask to match stack.')
                mask = resize(mask, (EDS_copy.shape[0], EDS_copy.shape[1]))
            EDS_copyout = domult(EDS_copy, mask)     
    else:
        print("Mask and BCF file are not compatible")
        return 
    if cubemask == True:
        EDS_cubemask = EDS_copyout.copy()
        EDS_cubemask = EDS_cubemask.swapaxes(0,2)
        EDS_cubemask = EDS_cubemask.swapaxes(1,2)
        tif.imsave('Cube_Masked_Stack.tif', EDS_cubemask)
    
    #spectrum of original and masked one
    s_original = np.sum(np.sum(EDS.data, axis=0), axis=0)
    s_masked = np.sum(np.sum(EDS_copyout, axis=0), axis=0)
    #marked graph
    mask_graph = np.sum(EDS_copyout, axis=2)
    #export the spectrum numpy to csv
    E = EDS.axes_manager['Energy'].axis
    S = np.stack((E, s_masked), axis = 1)
    np.savetxt(args.output+"_spectrum_masked.csv", np.stack((E, s_masked), axis=1), delimiter=",")
    #Create graph from spectrum
    plt.plot(S[:,0], S[:,1], label = 'Spectrum_Plot')
    plt.yscale('log')
    plt.ylabel('')
    plt.xlabel('eV')
    plt.savefig(args.output+'_Spectrum_Plot.png', dpi=120)
    plt.show()

if __name__ == "__main__":
    #arguments for command line parsing
    parser = argparse.ArgumentParser(description='makeSpectrum from a mask and bcf file.')
    parser.add_argument('--exportHAADF', '-e', action='store_true', help = 'Export the HAADF file to a tif.')
    parser.add_argument('stack', metavar='stack.bcf', type=str, help = 'Stack as a bcf file.')
    parser.add_argument('mask', nargs='?', metavar='mask.tif', type=str, help = 'Mask as a tif file')
    parser.add_argument('--output', '-o', action='store', type=str, default='noname', help='Ouput file name for CSV file, the default is built from the bcf and tif file names')
    parser.add_argument('--exportCUBE', action='store_true', help = 'Export the Cube stack to a tif.')
    # Add optional argument -o for output file
    args = parser.parse_args()
    if args.output == 'noname':
        if args.mask:
            args.output = f'{os.path.splitext(args.stack)[0]}_{os.path.splitext(args.mask)[1]}.csv'
        else: 
            args.output = f'{os.path.splitext(args.stack)[0]}.csv'
    #load the HAADF and EDS files from the stack
    HAADF, EDS = hs.load(args.stack)
    print(f'Stack dimensions are: {EDS.data.shape}.')
    
    #Save the HAADF tif
    if args.mask is None or args.exportHAADF:
        print(f'HAADF dimensions are: {HAADF.data.shape}.')
        HAADF.save('HAADF.tif')

    #If prompted, export a cube tif file of Stack
    if args.mask:
        mask = skimage.io.imread(args.mask)
        print(f'Mask dimensions are: {mask.shape}.')
        mask = (mask/np.max(mask))
        if args.exportCUBE:
            makeSpectrum(EDS, mask, cubemask = True)
        else:
            makeSpectrum(EDS, mask)

    if args.exportCUBE and args.mask is None:
        EDS_copy2 = EDS.data.copy()
        EDS_copy2 = EDS_copy2.swapaxes(0,2)
        EDS_copy2 = EDS_copy2.swapaxes(1,2)
        tif.imsave('Cube_Stack.tif', EDS_copy2)
