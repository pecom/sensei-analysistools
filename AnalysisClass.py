import numpy as np
from astropy.io import fits
from scipy import stats
from astropy import stats as astrostats
import matplotlib.pyplot as plt
from time import time
import pickle
import re
import os
import copy
import convertADU2e as convert
from scipy.ndimage import label
from scipy.ndimage import labeled_comprehension
from scipy import signal
import xml.etree.ElementTree as ET

# DataFiles Object holds the mask and science files and offers some modification of the masks.
# This is mostly focused on setting up the files in a convenient form for the Analysis object

class DataFiles:
    
    # Tiny useful lambda functions that are nice to have on hand
    get_minmax = lambda self, x : (x.min(), x.max())
    get_halobds = lambda self, mi, ma, r, low, up : (max(low, mi - r), min(up, ma + r + 1))
    nload = lambda self, k : np.load(k)['arr_0'] 
    
    # Generic way to get files. Given a file directory (filedir) and a Regex pattern
    # (patt) it will return the filenames of all the files that match.
    def get_files(self, filedir, patt):
        tempfilenames = []
        for fname in os.listdir(filedir):
            if re.match(r"%s" % patt, fname):
                tempfilenames.append(filedir + fname)
        return tempfilenames
    
    ########################################################
    # The following code blocks work for 2020 paper data and 2021 MINOS.
    # Generically they fetch the files and then order them based on the numbering
    # as defined by (fnum). This is a bit hard coded but it relies on consistent
    # file naming schemes more than anything else.
    # The files are in sfs_full (science files full) or mfs_full (mask files full)
    # and the file names are stored in sfs_names/mfs_names
    ########################################################
    


    # This function relies on running Daniel's analysis code first
    def get_samplefiles(self, filedir, name='valuesE_'): 
        fnames = np.array(self.get_files(filedir, name))
        fnum = np.array(list(map(lambda x : int(x.split('_')[2]), fnames)))
        fsort = np.argsort(fnum)
        forder = fnames[fsort]
        self.sfs_full = [self.nload(f) for f in forder]
        self.sfs_names = forder
        
    # This function relies on running Daniel's analysis code first
    def get_maskfiles(self, filedir, name='mask_0'):
        fnames = np.array(self.get_files(filedir, name))
        fnum = np.array(list(map(lambda x : int(x.split('_')[2]), fnames)))
        fsort = np.argsort(fnum)
        forder = fnames[fsort]
        self.mfs_full = [self.nload(f) for f in forder]
        self.mfs_names = forder
        

    # Given a list of file names, add data to procfits for a specificied quadrant
    def add_procs(self, procfiles):
        self.procfits = []
        for p in procfiles:
            with fits.open(p) as t1:
                self.procfits.append(t1[self.quadrant].data)
        self.proc_names = procfiles

    def add_masks(self, procfiles):
        self.maskfits = []
        for p in procfiles:
            with fits.open(p) as t1:
                self.maskfits.append(t1[self.quadrant].data)
        self.mask_names = procfiles

    def add_cal(self, cals):
        modquad = self.quadrant + 1
        self.cal_gains = []
        for cx in cals:
            gain = 0
            rx = ET.parse(cx).getroot()
            for rp in rx[0]:
                if 'num' in rp.attrib:
                    if int(rp.attrib['num']) == modquad:
                        gain = float(rp.attrib['gain'])
            if gain==0:
                print("Using default gain for quadrant %i of file %s" % (self.quadrant,cx))
                gain = float(rx[0][0].attrib['gain'])
            self.cal_gains.append(gain)
        self.cal_names = cals

    # Wrapper function to get proc files using regex search
    def get_fits(self, filedir, name='proc'):
        fnames = np.array(self.get_files(filedir, name))
        fnum = [int(g.split('_')[-1][:-5]) for g in fnames]
        procfiles = fnames[np.argsort(fnum)]
        self.add_procs(procfiles)
    # Wrapper function to get mask files using regex search
    def get_maskfits(self, filedir, name='mask_corr'):
        fnames = np.array(self.get_files(filedir, name))
        fnum = [int(g.split('_')[-1][:-5]) for g in fnames]
        procfiles = fnames[np.argsort(fnum)]
        self.add_masks(procfiles)
    # Wrapper function to get calibration files using regex search
    def get_calxml(self, filedir, name='cal_proc_corr'):
        cals = np.array(self.get_files(filedir, name))
        cnum = [int(cs.split('_')[-1][:3]) for cs in cals]
        cnames = cals[np.argsort(cnum)]
        self.add_cal(cnames)

    # Helper function to convert floats into electron counts
    def electronize(self, fdadu, bn, gd):
        el_data = np.zeros_like(fdadu)
        binfilt = fdadu < bn[-1]    
        gdfilt = fdadu >= bn[-1]
        el_data[binfilt] = np.digitize(fdadu[binfilt], bn, right=False) - 1
        el_data[gdfilt] = np.round(fdadu[gdfilt] - (gd - .5))
        return el_data
    
    # Convert fits files int o integer counts using electronize and parameters defined by binParams
    def proc_convert(self):
        self.sfs_full = []
        binned_list = [self.binParams["bottomCut"], self.binParams["zeroOneCut"],
                       self.binParams["oneTwoCut"], self.binParams["twoThreeCut"]]
        for g in self.procfits:
            adu0, _, off0 = convert.findConv(g, "", "", False, True)
            ef0 = self.electronize((g-off0)/adu0, binned_list, self.binParams["zeroOneCut"])
            
            ef0[ef0==-1] = 0
            self.sfs_full.append(ef0.astype(int))
        
    # Apply the calibration xml files to convert from procs to integer counts
    def cal_convert(self):
        self.sfs_full = []
        binned_list = [self.binParams["bottomCut"], self.binParams["zeroOneCut"],
                       self.binParams["oneTwoCut"], self.binParams["twoThreeCut"]]

        for g,gain in zip(self.procfits, self.cal_gains):
            ef0 = self.electronize(g/gain, binned_list, self.binParams["zeroOneCut"])
            ef0[ef0==-1] = 0
            self.sfs_full.append(ef0.astype(int))
        
    # Convert fits mask files to the normal masks (no real processing, just some relabelling)
    def mfit_convert(self):
        self.mfs_full = []
        for g in self.maskfits:
            self.mfs_full.append(g)
    
    # Some files have excess regions which we don't care about when counting so we can trim them 
    # to make our lives easier down the line.
    def trim_files(self, sf, mf, minx, maxx, miny, maxy):
        newsf = sf[minx:maxx, miny:maxy]
        newmf = mf[minx:maxx, miny:maxy]
        return newsf, newmf
    
    # Call trim_files on all the science (sfs_full) and mask (mfs_full) files
    # and store them in sfs and mfs
    def trim_all(self, minx, maxx, miny, maxy):
        sfs = []
        mfs = []
        for mf, sf in zip(self.mfs_full, self.sfs_full):
            nsf, nmf = self.trim_files(sf, mf, minx, maxx, miny, maxy)
            sfs.append(nsf)
            mfs.append(nmf)
        self.sfs = sfs
        self.mfs = mfs
        self.partial_expo = self.full_exp2d[minx:maxx, miny:maxy]
        

    ########################################################
    # Here are the functions that help modify the mask. In general
    # the boolean associated with the mask is passed in as an argument
    # which allows from flexibility down the line. Most of the functions
    # do directly modify the mask arrays in the code but will NOT
    # modify the npz files that they are loaded from.
    ########################################################


    # Generically remove a mask value from a list of mask files
    def remove_maskval(self, masks, edgeval):
        for m in masks:
            edgefilt = (m & edgeval).astype(bool)
            m[edgefilt] -= edgeval
        return masks
        
    # Add in an edge mask.
    def add_edgemask(self, masks, edgeval, rad):
        if rad > 0:
            for m in masks:
                m[:rad, :] |= edgeval
                m[:, :rad] |= edgeval
                m[-rad:, :] |= edgeval
                m[:, -rad:] |= edgeval
        return masks
    
    # Add a column mask
    def add_columnmask(self, masks, edgeval, rad, bothSides=True):
        if rad > 0:
            for m in masks:
                m[:rad, :] |= edgeval
                if bothSides:
                    m[-rad:, :] |= edgeval
        return masks
    
    # Add a column mask
    def add_rowmask(self, masks, edgeval, rad, bothSides=True):
        if rad > 0:
            for m in masks:
                m[:, :rad] |= edgeval
                if bothSides:
                    m[:, -rad:] |= edgeval
        return masks
    
    # Helper function to make circles for the Halo Radius mask
    def get_circ3(self, x, y, r, tfilt, xmin = 0, ymin=0, xmax = 443, ymax = 3072):
        xs = np.arange(x-r, x+r+1, 1)
        ys = np.arange(y-r, y+r+1, 1)
        xx, yy = np.meshgrid(xs, ys, indexing='ij')
        xfilt = (xx > xmin) * (xx < xmax)
        yfilt = (yy > ymin) * (yy < ymax)
        full_filt = xfilt * yfilt * tfilt
        return xx[full_filt],  yy[full_filt] 
    
    # Helper function to make circles for the Halo Radius mask
    def get_genfilt(self, r):
        xx,yy = np.mgrid[-r:r+1, -r:r+1]
        tf = ((xx**2 + yy**2) <= r**2)
        return tf

    def get_bleedfilt(self, r):
        ks = 2*r+1
        kale = np.zeros((ks,ks))
        kale[r:, r] = 1
        kale[r, r:] = 1
        return kale
    
    # Add a halo mask. Note that this one needs the science files (images) as an argument!
    def add_halomask(self, masks, images, haloval, rad):
        numero = np.arange(len(masks))
        imsize = images[0].shape
        tfilt = self.get_genfilt(rad)
        for m, im, n in zip(masks, images, numero):
            halo_pos = np.argwhere(im > 100)
            for h in halo_pos:
                m[self.get_circ3(*h, rad, tfilt, xmax=imsize[0], ymax=imsize[1])] |= haloval
        return masks

    # Add a halo mask using convolution to make life easier and faster!
    def conv_halomask(self, masks, images, haloval, rad):
        numero = np.arange(len(masks))
        imsize = images[0].shape
        tfilt = self.get_genfilt(rad).astype(int)
        for m, im, n in zip(masks, images, numero):
            halo_pos = (im > 100)
            conved_mask = np.round(signal.fftconvolve(halo_pos, tfilt, mode='same')).astype(bool)
            m[conved_mask] |= haloval
        return masks
    
    # Add a halo mask using convolution to make life easier and faster!
    def cluster_halomask(self, masks, images, haloval, rad):
        numero = np.arange(len(masks))
        imsize = images[0].shape
        tfilt = self.get_genfilt(rad).astype(int)
        for m, im, n in zip(masks, images, numero):
            [clusterID,numClu] = label(im, self.s)
            lbls = np.arange(numClu+1)
            energyTots = labeled_comprehension(im, clusterID, lbls, lambda x : np.sum(x), float, 0)
            good_energy = (energyTots >= 100)
            good_thresh = lbls[good_energy]
            good_mask = np.zeros_like(clusterID).astype(bool)
            for g in good_thresh:
                good_mask += (clusterID == g)
            good_mask = good_mask.astype(int) * 200

            halo_pos = (good_mask > 100)
            conved_mask = np.round(signal.fftconvolve(halo_pos, tfilt, mode='same')).astype(bool)
            m[conved_mask] |= haloval
        return masks

    # Add a bleed mask using convolution to make life easier and faster!
    # Rad should be 100 for 1e- analysis and 50 for 2e- analysis
    def conv_bleedmask(self, masks, images, bleedval, rad):
        numero = np.arange(len(masks))
        imsize = images[0].shape
        tfilt = self.get_bleedfilt(rad).astype(int)
        for m, im, n in zip(masks, images, numero):
            hee_pos = (im > 100)
            conved_mask = np.round(signal.fftconvolve(hee_pos, tfilt, mode='same')).astype(bool)
            m[conved_mask] |= bleedval
        return masks

    # Add a hotcolumn mask
    def add_hotcolmask(self, masks, hotcols, hcvalue, prescan=False ):
        # Prescan==True -> The hot columns INCLUDE the prescan region
        for m in masks:
            if prescan:
                hcs = hotcols - 8   # Probably should make this a generic value
            else:
                hcs = hotcols
            for hc in hcs:
                if hc > 0:
                    m[:, hc] |= hcvalue
        return masks

    # Add a hot pixel mask
    def add_hotpixel(self, masks, hotpix, hpvalue):
        for m in masks:
            for hp in hotpix:
                m[hp[1], hp[0]] |= hpvalue
        return masks


    # Convenience function to update (remove and add new) edge mask
    def update_edge(self, mflag, rad):
        _ = self.remove_maskval(self.mfs, mflag)
        _ = self.add_edgemask(self.mfs, mflag, rad)
    
    # Convenience function to update (remove and add new) halo mask
    def update_halo(self, mflag, rad):
        _ = self.remove_maskval(self.mfs, mflag)
        _ = self.add_halomask(self.mfs, self.sfs, mflag, rad) 
        
    # Convenience function to update (remove and add new) halo mask using convolve method
    def update_convhalo(self, mflag, rad):
        _ = self.remove_maskval(self.mfs, mflag)
        _ = self.conv_halomask(self.mfs, self.sfs, mflag, rad) 
        
    # Convenience function to update (remove and add new) bleed mask using convolve method
    def update_convbleed(self, mflag, rad):
        _ = self.remove_maskval(self.mfs, mflag)
        _ = self.conv_bleedmask(self.mfs, self.sfs, mflag, rad) 
    # Given the bad flags  (bfs) we turn the mask files (mfs)
    # into boolean arrays that are "inverse masks" such that
    # True means we WANT to count the pixel and False is we
    # do NOT want to count the pixel. 
    def masks_2_flags(self, mfs, bfs):
        flagsum = int(np.sum([self.maskFlags[b] for b in bfs]))
        # True means we want to count the value of the pixel
        # False means we DO NOT want to count the value of the pixel
        flaggedmasks = [~((mf & flagsum).astype(bool)) for mf in mfs]
        self.flagged = flaggedmasks
        return flaggedmasks
    

    # Non-negligible readout time leads to varying exposure time per pixel.
    # Calculates the proper full exposure time.
    def make_exp(self, dw, dh, expBase):
        self.dHeight = dh
        self.dWidth = dw
        self.pix_inquad = dw*dh
        self.exposureBase = expBase
        self.fullExposureTimes = np.array(np.fromfunction(
                lambda i, j:(self.exposureBase+i*self.singleReadTime),
                (self.pix_inquad,1),dtype=int))

        self.full_exp2d = np.array(np.fromfunction(
            lambda i, j:(self.exposureBase+i*self.singleReadTime),
            (self.pix_inquad,1),dtype=int)).reshape((dw, dh))
        
    # Set bin parameters
    def set_bp(self, bp):
        self.binParams = bp
        
    # Convenience function with some parameters from 2020 Paper data
    def set_2020(self):
        self.singleReadTime = 42.825 * 1e-6 * 300
        self.binParams = {
                            "zeroOneCut": .63,#51,#51,#.63, #0.95 #cutoff (in pixels) between 0e and 1e bin
                            "oneTwoCut": 1.63, #1.8 #cutoff (in pixels) between 1e and 2e bin
                            "twoThreeCut": 2.63, #cutoff (in pixels) between 2e and 3e bin
                            "bottomCut": -1.2, #minimum value that we will consider a "valid" 0-e event
                            "countCut": 0.85 #Cut to use for statistical conversion (0.85 was found to be optimal)
                        }
        self.dHeight = 3100
        self.dWidth = 470
        self.quad_mass = self.dHeight * self.dWidth * self.grams_per_pixel
        
    # Convenience function with parameters from 2021 Minos data
    def set_minos3(self):
        self.singleReadTime = 0.01444
        self.dHeight = 3200
        self.dWidth = 520
        self.binParams = {
                    "zeroOneCut": .7,#51,#51,#.63, #0.95 #cutoff (in pixels) between 0e and 1e bin
                    "oneTwoCut": 1.63, #1.8 #cutoff (in pixels) between 1e and 2e bin
                    "twoThreeCut": 2.63, #cutoff (in pixels) between 2e and 3e bin
                    "bottomCut": -1.2, #minimum value that we will consider a "valid" 0-e event
                    "countCut": 0.85, #Cut to use for statistical conversion (0.85 was found to be optimal)
                    "fractional2s" : True #Shoudl we calcualte the number of 2s passing the cut, or add the probability?
                }
        self.quad_mass = self.dHeight * self.dWidth * self.grams_per_pixel
        self.scharge = 7.027e-5
        
    def __init__(self, quad, ecount):
        self.ecount = ecount
        self.updateMasks = False
        self.quadrant = quad
        # quadrants = [0,1,2]
        self.s = np.array([[1, 1, 1], [1,1,1], [1,1,1]])
        self.e2ev = 3.8
        self.spurious_charge = 1.594*10**(-4) # e/pix/day from paper
        self.scharge = 0.0001664 # e/pix from 2020 code
        self.exposureBase = 20*3600 # seconds
        self.singleReadTime = 42.825*1e-6*300
        self.sig_lvl = 2
        self.dHeight = 3100
        self.dWidth = 470
        self.pix_inquad_daniel = 470*3100
        self.fullExposureTimes = np.array(np.fromfunction(
                lambda i, j:(self.exposureBase+i*self.singleReadTime),
                (self.pix_inquad_daniel,1),dtype=int))
        self.full_exp2d = np.array(np.fromfunction(
            lambda i, j:(self.exposureBase+i*self.singleReadTime),
            (self.pix_inquad_daniel,1),dtype=int)).reshape((3100, 470)).T
        self.grams_per_pixel = 3.537e-7
        self.seconds_in_day = 24*60*60
        self.ps2gd = (1/self.seconds_in_day * self.grams_per_pixel)
        self.binParams = {
                            "zeroOneCut": .63,#51,#51,#.63, #0.95 #cutoff (in pixels) between 0e and 1e bin
                            "oneTwoCut": 1.63, #1.8 #cutoff (in pixels) between 1e and 2e bin
                            "twoThreeCut": 2.63, #cutoff (in pixels) between 2e and 3e bin
                            "bottomCut": -1.2, #minimum value that we will consider a "valid" 0-e event
                            "countCut": 0.85 #Cut to use for statistical conversion (0.85 was found to be optimal)
                        }
        self.maskFlags= {
                            "neighborM" : 1,
                            "diagNeighM" : 1,
                            "hasHitM" : 2,
                            "bleedM" : 4,
                            "haloM" : 8,
                            "crosstalkM" : 16,
                            "noisyM": 32,
                            "edgeM" : 64, 
                            "serialM" : 128,
                            "clusterM": 256,
                            "badPixM" : 512,
                            "badColM": 1024,
                            "looseCluM" : 2048,
                            "extendBleedM": 4096,
                            "lowECluM" : 8192,
                            "fullWellM": 16384,
                            "badClusterM": 32768
                        }

# Analysis object is where most of the "counting" functions go.
# Once we have formatted our raw data, the rest of any analysis gets stuck in here
class Analysis:
    
    # Label the clusters and get energy of each cluster
    def labeled_energy(self, files):
        s = np.array([[1, 1, 1], [1,1,1], [1,1,1]])
        all_energy = []
        for sampmap in files:
            [clusterID,numClu] = label(sampmap, s)
            lbls = np.arange(numClu+1)
            energyTots = labeled_comprehension(sampmap, clusterID, lbls, lambda x : np.sum(x)*self.df.e2ev, float, 0)
            all_energy.append(energyTots)
        return all_energy
    
    # Naive search looking for Ne- (count) in 1 pixel
    def one_pix_search(self, files, masks, count):
        return np.array([np.sum(im[mk] == count) for im,mk in zip(files, masks)])
    
    # Slightly smarter search looking for Ne- (count) in a contiguous cluster
    def cluster_search(self, files, masks, count, s=np.array([[1, 1, 1], [1,1,1], [1,1,1]])):
        flen = len(files)
        subcounts = np.zeros((count, flen))
        for sampmap, fmask, nid in zip(files, masks, np.arange(flen)):
            # Standard use label to identify clusters
            [clusterID,numClu] = label(sampmap, s)
            lbls = np.arange(numClu+1)
            clusterTots = labeled_comprehension(sampmap, clusterID, lbls, np.sum, int, 0) # Get the number of electrons in each cluster
            nclusts = np.where(clusterTots==count)[0] # Where do we have "count" # of electrons
            cleanclusts = nclusts[[~np.any(fmask[(clusterID==tc)] == False) for tc in nclusts]]
            multmaps = np.array([(clusterID==tc) * fmask for tc in cleanclusts])
            for k in range(count):
                kpixarr = np.array([np.sum(mm)==(k+1) for mm in multmaps])
                subcounts[k, nid] = np.sum(kpixarr)
        return subcounts, np.sum(subcounts) 

    def channel_search(self, files, masks):
        flen = len(files)
        s = np.ones((3,3))
        svert = np.array([[0,1,0],[0,1,0],[0,1,0]])
        shori = np.array([[0,0,0], [1,1,1], [0,0,0]])
        sdiag = np.array([[1,0,1],[0,1,0],[1,0,1]])
        ssolo = -1*np.ones((3,3))
        ssolo[1,1] = 1
        
        stypes = [svert, shori, sdiag, ssolo]
        sfin = [2,2,2,1] # What should the convolution give us?
        snames = ["Vertical", "Horizontal", "Diagonal", "Solo"]

        schannel = np.zeros((flen, 5))
        for sampmap, fmask, nid in zip(files, masks, np.arange(flen)):
            # Standard use label to identify clusters
            [clusterID,numClu] = label(sampmap, s)
            lbls = np.arange(numClu+1)
            clusterTots = labeled_comprehension(sampmap, clusterID, lbls, np.sum, int, 0) # Get the number of electrons in each cluster
            nclusts = np.where(clusterTots==2)[0] # Where do we have "count" # of electrons
            cleanclusts = nclusts[[~np.any(fmask[(clusterID==tc)] == False) for tc in nclusts]]
            multmaps = np.array([(clusterID==tc) * fmask for tc in cleanclusts])
            if len(multmaps) == 0:
                continue
            fmap = np.zeros_like(multmaps[0])
            for mm in multmaps:
                if np.sum(mm)>0:
                    schannel[nid, 0] +=  1
                fmap += mm
            for i,st in enumerate(stypes):
                conved_fmap = signal.convolve2d(fmap, st, mode='same')
                if sfin[i]==1:
                    channel_count = np.sum(conved_fmap == 1)
                else:
                    channel_count = np.sum(conved_fmap == 2)/2
                schannel[nid, i+1] += channel_count
        return schannel 
            
    # Add a halo mask to each pixel in a cluster
    def he_boolgrid(self, cluID, l, radius, include=True):
        clust_pixs = np.where(cluID==l)
        zipped_clust = np.array(list(zip(*clust_pixs)))

        xmin, xmax = self.df.get_minmax(clust_pixs[0])
        ymin, ymax = self.df.get_minmax(clust_pixs[1])
        
        im_xmax = self.df.sfs[0].shape[0]
        im_ymax = self.df.sfs[0].shape[1]

        halo_xmin, halo_xmax = self.df.get_halobds(xmin, xmax, radius, 0, im_xmax)
        halo_ymin, halo_ymax = self.df.get_halobds(ymin, ymax, radius, 0, im_ymax)

    #     halo_xmin, halo_xmax = get_halobds(xmin, xmax, radius, 0, 470)
    #     halo_ymin, halo_ymax = get_halobds(ymin, ymax, radius, 0, 3100)


        poss_grid = np.mgrid[halo_xmin:halo_xmax, halo_ymin:halo_ymax]
        bool_grid = np.zeros_like(poss_grid[0], dtype=bool)

        off_origin = np.array([halo_xmin, halo_ymin])

        tfil = self.df.get_genfilt(radius)

        for z in zipped_clust:
            circx, circy = self.df.get_circ3(*z, radius, tfil, xmax=im_xmax, ymax=im_ymax)
            bool_grid[circx - off_origin[0], circy-off_origin[1]] = True

        # Do we INCLUDE the cluster in the MASK
        # aka True -> do NOT count pixels in the cluster 
        if not (include):
            for z in zipped_clust:
                bool_grid[z[0] - off_origin[0], z[1] - off_origin[1]] = False
        return bool_grid, poss_grid
    
    # Create halo radius plot with extra binning for energy of cluster.
    # If -1 is used as a right edge for a bin it is treated as +infinity
    # Different return formatting
    def conv_cherenkov_binned(self, im, mk, bins, rads, fixrad, include=True):
        
        pixels = []
        expos = []
        counts = []
        
        sampmask = copy.deepcopy(mk)
        [clusterID,numClu] = label(im, self.df.s)
        lbls = np.arange(numClu+1)
        energyTots = labeled_comprehension(im, clusterID, lbls, lambda x : np.sum(x)*self.df.e2ev, float, 0)
        haloVal = self.df.maskFlags['haloM']
        edgeVal = self.df.maskFlags['edgeM']

        
        # bins should be in increasing order!
        for i in range(len(bins)-1):
            bin_pixels = []
            bin_expos = []
            bin_counts = []

            _ = self.df.remove_maskval([sampmask], haloVal)


            firstRun = True
            if bins[i+1]== -1:
                good_energy = (energyTots >= bins[i])
            else:
                good_energy = (energyTots < bins[i+1]) * (energyTots >= bins[i])

            if bins[-1] ==-1:
                full_range = energyTots >= bins[0]
            else:
                full_range = (energyTots >= bins[0]) * (energyTots < bins[-1])

            else_energy = full_range ^ good_energy
            good_thresh = lbls[good_energy]
            else_thresh = lbls[else_energy]
            # print("Number in good bin:", len(good_thresh))
            
            good_mask = np.zeros_like(clusterID).astype(bool)
            else_mask = np.zeros_like(clusterID).astype(bool)
            for g in good_thresh:
                good_mask += (clusterID == g)
            # print("Total cluster pixels:", np.sum(good_mask))
            # print("Cluster pixels:", [(x,y) for x,y in zip(*np.where(good_mask))])
            for e in else_thresh:
                else_mask += (clusterID == e)
            good_mask = good_mask.astype(int) * 200
            else_mask = else_mask.astype(int) * 200

            for r in rads:
                _ = self.df.remove_maskval([sampmask], haloVal)
                _ = self.df.remove_maskval([sampmask], edgeVal)
                hrad = r
                lrad = fixrad
                temp = np.zeros_like(clusterID)

                goodHalo = self.df.conv_halomask([temp], [good_mask], haloVal, hrad)[0].astype(bool)
                if firstRun:
                    temp = np.zeros_like(clusterID)
                    elseHalo = self.df.conv_halomask([temp], [else_mask], haloVal, lrad)[0].astype(bool)
                sampmask[goodHalo] |= haloVal
                sampmask[elseHalo] |= haloVal

                _ = self.df.add_edgemask([sampmask], edgeVal, r)

                edge_pixs = np.sum((sampmask & edgeVal) == edgeVal)
                halo_pixs = np.sum((sampmask & haloVal) == haloVal)
                rest = np.size(sampmask)
                rest_sub  = rest - edge_pixs - halo_pixs
                # print(f"The edge {edge_pixs}, the halo {halo_pixs}, rest: {rest} and subtracted {rest_sub}")



                sampflag = self.df.masks_2_flags([sampmask], self.flags)
                f2 = self.flags.copy()
                f2.discard('neighborM')
                sampflag_2 = self.df.masks_2_flags([sampmask], f2)


                exposure = np.ma.masked_array(self.df.partial_expo, ~(sampflag[0]).flatten(order='F'))
                exposure_2 = np.ma.masked_array(self.df.partial_expo, ~(sampflag_2[0]).flatten(order='F'))
                # if r==10:
                #     print("Mask sum: ", np.sum(sampflag))
                expo_size = np.sum(exposure)
                expo2_size = np.sum(exposure_2)
                expos_obj = (expo_size, expo2_size)

                events = self.one_pix_search([im], sampflag, 1)[0]
                events_2 = self.one_pix_search([im], sampflag_2, 2)[0]
                kale, two_e = self.cluster_search([im], sampflag_2, 2)
                _, three_e = self.cluster_search([im], sampflag_2, 3)
                events_obj = (events,events_2, two_e, three_e)
                
                pix1 = np.sum(sampflag)
                pix2 = np.sum(sampflag_2)
                pixs_obj = (pix1, pix2)

                firstRun = False
                bin_pixels.append(pixs_obj)
                bin_counts.append(events_obj)
                bin_expos.append(expos_obj)
            pixels.append(bin_pixels)
            counts.append(bin_counts)
            expos.append(bin_expos)
        
        return pixels, counts, expos
    
    # Create halo radius plot with extra binning for energy of cluster.
    # Only 2 bins allowed!
    def conv_cherenkov_low(self, im, mk, bins, rads, lowrad, include=True, verbose=False):
        
        pixels = []
        expos = []
        counts = []
        
        sampmask = copy.deepcopy(mk)
        [clusterID,numClu] = label(im, self.df.s)
        lbls = np.arange(numClu+1)
        energyTots = labeled_comprehension(im, clusterID, lbls, lambda x : np.sum(x)*self.df.e2ev, float, 0)
        haloVal = self.df.maskFlags['haloM']
        edgeVal = self.df.maskFlags['edgeM']
        bleedVal = self.df.maskFlags['bleedM']

        bin_pixels = []
        bin_expos = []
        bin_counts = []

        _ = self.df.remove_maskval([sampmask], haloVal)


        low_energy = (energyTots >= bins[0]) * (energyTots <= bins[1])
        full_range = energyTots >= bins[0]

        high_energy = full_range ^ low_energy
        low_thresh = lbls[low_energy]
        high_thresh = lbls[high_energy]
        
        low_mask = np.zeros_like(clusterID).astype(bool)
        high_mask = np.zeros_like(clusterID).astype(bool)
        for g in low_thresh:
            low_mask += (clusterID == g)
        for e in high_thresh:
            high_mask += (clusterID == e)
        low_mask = low_mask.astype(int) * 200
        high_mask = high_mask.astype(int) * 200

        firstRun = True
        for r in rads:
            _ = self.df.remove_maskval([sampmask], haloVal)
            _ = self.df.remove_maskval([sampmask], edgeVal)
            hrad = r
            lrad = lowrad
            if firstRun:
                temp = np.zeros_like(clusterID)
                lowHalo = self.df.conv_halomask([temp], [low_mask], haloVal, lrad)[0].astype(bool)
                firstRun = False

            temp = np.zeros_like(clusterID)
            highHalo = self.df.conv_halomask([temp], [high_mask], haloVal, hrad)[0].astype(bool)
            sampmask[lowHalo] |= haloVal
            sampmask[highHalo] |= haloVal

            _ = self.df.add_edgemask([sampmask], edgeVal, r)
            _ = self.df.remove_maskval([sampmask], bleedVal)
            _ = self.df.conv_bleedmask([sampmask], [im], bleedVal, 100)
            sampflag = self.df.masks_2_flags([sampmask], self.flags)

            _ = self.df.remove_maskval([sampmask], bleedVal)
            _ = self.df.conv_bleedmask([sampmask], [im], bleedVal, 50)
            f2 = self.flags.copy()
            f2.discard('neighborM')
            sampflag_2 = self.df.masks_2_flags([sampmask], f2)

            if verbose:
                gmone = im*sampflag[0]
                print('Where gmone:', np.where(gmone==1))
                # print("mask:", sampmask[41, 39], sampmask[39,41])


            exposure = np.ma.masked_array(self.df.partial_expo, ~(sampflag[0]).flatten(order='F'))
            exposure_2 = np.ma.masked_array(self.df.partial_expo, ~(sampflag_2[0]).flatten(order='F'))
            expo_size = np.sum(exposure)
            expo2_size = np.sum(exposure_2)
            expos_obj = (expo_size, expo2_size)

            events = self.one_pix_search([im], sampflag, 1)[0]
            events_2 = self.one_pix_search([im], sampflag_2, 2)[0]
            _, two_e = self.cluster_search([im], sampflag_2, 2)
            _, three_e = self.cluster_search([im], sampflag_2, 3)
            events_obj = (events,0, two_e, three_e)
            
            pix1 = np.sum(sampflag)
            pix2 = np.sum(sampflag_2)
            pixs_obj = (pix1, pix2)


            bin_pixels.append(pixs_obj)
            bin_counts.append(events_obj)
            bin_expos.append(expos_obj)
        pixels.append(bin_pixels)
        counts.append(bin_counts)
        expos.append(bin_expos)
        
        return pixels, counts, expos

    def cluster_halos(self, im, mk, rads, ener_thresh=380, include=True, verbose=False):
        
        pixels = []
        expos = []
        counts = []
        
        sampmask = copy.deepcopy(mk)
        [clusterID,numClu] = label(im, self.df.s)
        lbls = np.arange(numClu+1)
        energyTots = labeled_comprehension(im, clusterID, lbls, lambda x : np.sum(x)*self.df.e2ev, float, 0)
        haloVal = self.df.maskFlags['haloM']
        edgeVal = self.df.maskFlags['edgeM']
        bleedVal = self.df.maskFlags['bleedM']

        _ = self.df.remove_maskval([sampmask], haloVal)

        good_energy = (energyTots >= ener_thresh)

        good_thresh = lbls[good_energy]
        
        good_mask = np.zeros_like(clusterID).astype(bool)
        for g in good_thresh:
            good_mask += (clusterID == g)
        good_mask = good_mask.astype(int) * 200

        for r in rads:
            _ = self.df.remove_maskval([sampmask], haloVal)
            _ = self.df.remove_maskval([sampmask], edgeVal)
            _ = self.df.remove_maskval([sampmask], bleedVal)
            hrad = r
            temp = np.zeros_like(clusterID)

            goodHalo = self.df.conv_halomask([temp], [good_mask], haloVal, hrad)[0].astype(bool)
            sampmask[goodHalo] |= haloVal

            _ = self.df.add_edgemask([sampmask], edgeVal, r)

            _ = self.df.remove_maskval([sampmask], bleedVal)
            _ = self.df.conv_bleedmask([sampmask], [im], bleedVal, 100)
            sampflag = self.df.masks_2_flags([sampmask], self.flags)

            _ = self.df.remove_maskval([sampmask], bleedVal)
            _ = self.df.conv_bleedmask([sampmask], [im], bleedVal, 50)
            f2 = self.flags.copy()
            f2.discard('neighborM')
            sampflag_2 = self.df.masks_2_flags([sampmask], f2)

            exposure = np.ma.masked_array(self.df.partial_expo, ~(sampflag[0]).flatten(order='F'))
            exposure_2 = np.ma.masked_array(self.df.partial_expo, ~(sampflag_2[0]).flatten(order='F'))

            expo_size = np.sum(exposure)
            expo2_size = np.sum(exposure_2)
            expos_obj = (expo_size, expo2_size)

            shoriz = np.array([[0,0,0],[1,1,1],[0,0,0]])
            svert = np.array([[0,1,0],[0,1,0],[0,1,0]])
            sdiag = np.array([[1,0,1],[0,1,0],[1,0,1]])
            ssolo = np.array([[0,0,0],[0,1,0],[0,0,0]])
            events = self.one_pix_search([im], sampflag, 1)[0]
            
            if verbose:
                gmone = im*sampflag[0]
                print('Where gmone:', np.where(gmone==1))
                print("mask:", sampmask[41, 39], sampmask[39,41])

            events_2 = self.one_pix_search([im], sampflag_2, 2)[0]
            kale, two_e = self.cluster_search([im], sampflag_2, 2)
            _, two_eh = self.cluster_search([im], sampflag_2, 2, shoriz)
            _, two_ev = self.cluster_search([im], sampflag_2, 2, svert)
            _, two_ed = self.cluster_search([im], sampflag_2, 2, sdiag)
            _, two_es = self.cluster_search([im], sampflag_2, 2, ssolo)
            _, three_e = self.cluster_search([im], sampflag_2, 3)
            events_obj = (events,events_2, two_e, two_eh, two_ev, two_ed, two_es, three_e)
            
            pix1 = np.sum(sampflag)
            pix2 = np.sum(sampflag_2)
            pixs_obj = (pix1, pix2)

            firstRun = False
            pixels.append(pixs_obj)
            counts.append(events_obj)
            expos.append(expos_obj)
        
        return pixels, counts, expos
    # Estimate errors on counts. For high N we can approximate with 1/sqrt(N)
    # but low N has to use poisson error estimate.
    # Returns the Delta_lower and Delta_upper bound such that
    # Delta_lower + data = lower bound
    def create_errors(self, counts, sig):
        count_errs = np.zeros((2, len(counts)))
        for i, n in enumerate(counts):
            if n < 30:
                ci = np.abs(astrostats.poisson_conf_interval(n, interval='frequentist-confidence',sigma=sig) - n)
            else:
                ci = [np.sqrt(n)*sig]*2
            count_errs[0,i] = ci[0]
            count_errs[1,i] = ci[1]
        return count_errs

    def poiss_upperlimit(count, perc=.9):
        # Gamma function for Poisson upper limit calculations
        gamma = lambda k, ts: ts**k * np.exp(-ts)
        tmax = 50
        deltaT = .01
        ts1 = np.arange(0,tmax, deltaT)
        g1 = gamma(count, ts1)
        rhs = np.cumsum(g1 * deltaT)
        lhs = perc * np.math.factorial(count)
        h1 = ts1[np.argmin(np.abs(rhs - lhs)) + 1]
        return h1

    # Calculate the upper limits given counts and percentage
    def create_upperlimit(self, counts, perc=.90):
        count_UL = np.zeros(len(counts))
        norm_perc = perc + (1-perc)/2
        sig = stats.norm.ppf(norm_perc, 0, 1)

        if self.poisson is not None:
            pass
        else:
            poiss = {}
            for i in range(31):
                poiss[i] = poiss_upperlimit(i)
            self.poisson = poiss
        for i,k in enumerate(counts):
            if k > 30:
                h1 = ((sig + np.sqrt(sig**2 + 4*k))/2)**2
            else:
                h1 = poiss[k]
            count_UL[i] = h1

        return count_UL
        
    
    # Count the number of 1e- events in a set of rows
    def proper_row_rate(self, ims, mks, binsize, start=1, stop=0):
        if stop==0:
            maxrow = ims[0].shape[1]
        else:
            maxrow = stop
        minrows = np.arange(start, maxrow, binsize)
        maxrows = np.arange(binsize+start, maxrow+binsize, binsize)
        kcount = np.arange(len(minrows))
        
        expos = []
        counts = []
        pixs = []
        
        for im,mk in zip(ims, mks):
            ex_i = np.zeros_like(minrows)
            count_i = np.zeros_like(ex_i)
            pix_i = np.zeros_like(ex_i)
            for bmin, bmax, k in zip(minrows, maxrows, kcount):
                subset_im = im[:, bmin:bmax]
                subset_mk = mk[:, bmin:bmax]
                subset_flag = self.df.masks_2_flags([subset_mk], self.flags)
                subset_expo = self.df.partial_expo[:, bmin:bmax]

                exposure = np.sum(subset_expo * subset_flag)
                ex_i[k] = exposure
                count_i[k] = self.one_pix_search([subset_im], subset_flag, 1)
                pix_i[k] = np.sum(subset_flag)
            expos.append(ex_i)
            counts.append(count_i)
            pixs.append(pix_i)
        return np.array(pixs), np.array(counts), np.array(expos), minrows
    
    
    # Count the number of 1e- events in a set of columns
    def proper_col_rate(self, ims, mks, binsize, start=1, stop=0):
        if stop==0:
            maxcol = ims[0].shape[0]
        else:
            maxcol = stop
        mincols = np.arange(start, maxcol, binsize)
        maxcols = np.arange(binsize+start, maxcol+binsize, binsize)
        kcount = np.arange(len(mincols))
        
        expos = []
        counts = []
        pixs = []
        
        for im,mk in zip(ims, mks):
            ex_i = np.zeros_like(mincols)
            count_i = np.zeros_like(ex_i)
            pix_i = np.zeros_like(ex_i)
            for bmin, bmax, k in zip(mincols, maxcols, kcount):
                subset_im = im[bmin:bmax, :]
                subset_mk = mk[bmin:bmax, :]
                subset_flag = self.df.masks_2_flags([subset_mk], self.flags)
                subset_expo = self.df.partial_expo[bmin:bmax, :]

                exposure = np.sum(subset_expo * subset_flag)
                ex_i[k] = exposure
                count_i[k] = self.one_pix_search([subset_im], subset_flag, 1)
                pix_i[k] = np.sum(subset_flag)
            expos.append(ex_i)
            counts.append(count_i)
            pixs.append(pix_i)
        return np.array(pixs), np.array(counts), np.array(expos), mincols
    
    # Count the number of 1e- left after varying the halo mask
    def redo_haloplot2(self, im, mk, rads, ecount, minE=380, zero_row = True, include=False):

        sampmask = copy.deepcopy(mk)
        haloVal = self.df.maskFlags['haloM']
        edgeVal = self.df.maskFlags['edgeM']

        _ = self.df.remove_maskval([sampmask], haloVal)

        rkeys = []
        sflags = []
        pixsize = []
        expo = []
        counts = []
        firstRun = False

        if zero_row:
            im[0,:] = 0

        for r in rads:
            self.df.remove_maskval([mk], haloVal)
            self.df.add_halomask([mk], [im],  haloVal, r)
            self.df.update_edge(edgeVal, r)

            sampflag = self.df.masks_2_flags([mk], self.flags)
            exposure = np.ma.masked_array(self.df.partial_expo, ~(sampflag[0]).flatten(order='F'))
            expo_size = exposure.sum()
            events = self.one_pix_search([im], sampflag, 1)

            firstRun = True
            rkeys.append(r)
            sflags.append(sampflag)
            pixsize.append(np.sum(sampflag))
            counts.append(events)
            expo.append(expo_size)

        return rkeys, sflags, pixsize, counts, expo
    
    # Minor formatting to make the halo plotting easier
    def gather_redo_haloplot(self, rads):
        count_data = []
        expo_data = []
        pix_data = []
        for s, m, n in zip(self.df.sfs, self.df.mfs, range(len(self.df.sfs))):
            _, _, ps, c, ex = self.redo_haloplot2(s, m, rads, 1)
            count_data.append(c)
            expo_data.append(ex)
            pix_data.append(ps)
            print("Done with", n)
        return count_data, expo_data, pix_data
    
    # Helper function to correct for exposure dependent charage
    def correct_charges(self, expo, count):
        sc = self.df.scharge
        newcount = count - expo*sc
        return newcount
    
    def __init__(self, q, e, bf):
        self.flags = bf
        dfiles = DataFiles(q, e)
        self.df = dfiles
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
