import AnalysisClass as ac
from AnalysisClass import Analysis
from AnalysisClass import DataFiles
import numpy as np
import convertADU2e as convert
import sys
import time


badFlags = { 'badColM', 'badPixM', 'bleedM', 'crosstalkM', 'edgeM',
             'extendBleedM', 'fullWellM', 'haloM',
             'lowECluM', 'neighborM', 'noisyM', 'serialM'
}


badFlags_2e = { 'badColM', 'badPixM', 'bleedM', 'crosstalkM', 'edgeM',
             'extendBleedM', 'fullWellM', 'haloM', 
             'lowECluM', 'noisyM', 'serialM'
}



quads = [0,1,2,3]
quads = [0]
minos_halorad = []
# pix_sizes = np.arange(10, 110, 5)
pix_sizes = np.arange(0, 10, 1)
pix_sizes = [2,5,10]
# pix_sizes = [10]
ccdnum = int(sys.argv[1])
enum = int(sys.argv[2])
lowfix = int(sys.argv[3])

prfs = ['/data/users/meeg/minos3/proc_corr_proc_skp_moduleC40_41-ssc16_17-lta20_60_TEMP135K-run1_NROW520_NBINROW1_NCOL3200_NBINCOL1_EXPOSURE72000_CLEAR1800_1_250.fits']
#        '/data/users/meeg/minos3/proc_corr_proc_skp_moduleC40_41-ssc16_17-lta20_60_TEMP135K-run1_NROW520_NBINROW1_NCOL3200_NBINCOL1_EXPOSURE72000_CLEAR1800_1_255.fits',
#        '/data/users/meeg/minos3/proc_corr_proc_skp_moduleC40_41-ssc16_17-lta20_60_TEMP135K-run1_NROW520_NBINROW1_NCOL3200_NBINCOL1_EXPOSURE72000_CLEAR1800_1_260.fits',
#        '/data/users/meeg/minos3/proc_corr_proc_skp_moduleC40_41-ssc16_17-lta20_60_TEMP135K-run1_NROW520_NBINROW1_NCOL3200_NBINCOL1_EXPOSURE72000_CLEAR1800_1_265.fits']
        # '/data/users/meeg/minos3/proc_corr_proc_skp_moduleC40_41-ssc16_17-lta20_60_TEMP135K-run20_NROW520_NBINROW1_NCOL3200_NBINCOL1_EXPOSURE0_CLEAR1800_1_233.fits']


mrfs = ['/data/users/meeg/minos3/cuts_new/mask_corr_proc_skp_moduleC40_41-ssc16_17-lta20_60_TEMP135K-run1_NROW520_NBINROW1_NCOL3200_NBINCOL1_EXPOSURE72000_CLEAR1800_1_250.fits']
#        '/data/users/meeg/minos3/cuts_new/mask_corr_proc_skp_moduleC40_41-ssc16_17-lta20_60_TEMP135K-run1_NROW520_NBINROW1_NCOL3200_NBINCOL1_EXPOSURE72000_CLEAR1800_1_255.fits',
#        '/data/users/meeg/minos3/cuts_new/mask_corr_proc_skp_moduleC40_41-ssc16_17-lta20_60_TEMP135K-run1_NROW520_NBINROW1_NCOL3200_NBINCOL1_EXPOSURE72000_CLEAR1800_1_260.fits',
#        '/data/users/meeg/minos3/cuts_new/mask_corr_proc_skp_moduleC40_41-ssc16_17-lta20_60_TEMP135K-run1_NROW520_NBINROW1_NCOL3200_NBINCOL1_EXPOSURE72000_CLEAR1800_1_265.fits']
        # '/data/users/meeg/minos3/cuts_new/mask_corr_proc_skp_moduleC40_41-ssc16_17-lta20_60_TEMP135K-run20_NROW520_NBINROW1_NCOL3200_NBINCOL1_EXPOSURE0_CLEAR1800_1_233.fits']

hotcols = np.load(f'../data/badcols_{ccdnum}.npy', allow_pickle=True)
hotpix = np.load(f'../data/badpix_{ccdnum}.npy', allow_pickle=True)

data_dir = '/data/users/meeg/minos3/'
print("running on ccd: %i and exposure: %i" % (ccdnum, enum))

for q in quads:
    cr = ac.Analysis(q,  1, badFlags)
    minos_halorad.append(cr)

hpixval = minos_halorad[0].df.maskFlags['badPixM']
hcolval = minos_halorad[0].df.maskFlags['badColM']

for q, tr in zip(quads, minos_halorad):
    print("ON quadrant", q)

    newBp = {
        "zeroOneCut": .7,#51,#51,#.63, #0.95 #cutoff (in pixels) between 0e and 1e bin
        "oneTwoCut": 1.63, #1.8 #cutoff (in pixels) between 1e and 2e bin
        "twoThreeCut": 2.63, #cutoff (in pixels) between 2e and 3e bin
        "bottomCut": -1.2, #minimum value that we will consider a "valid" 0-e event
        "countCut": 0.85, #Cut to use for statistical conversion (0.85 was found to be optimal)
        "fractional2s" : True #Shoudl we calcualte the number of 2s passing the cut, or add the probability?
    }
    tr.df.set_bp(newBp)

    # prf = 'proc_corr_proc_skp.*EXPOSURE%i.*_%i_.*.fits' % (enum, ccdnum)
    # mrf = 'mask_corr_proc_skp.*EXPOSURE%i.*_%i_.*.fits' % (enum, ccdnum)
    # tr.df.get_fits('/data/users/meeg/minos3/', prf)
    # tr.df.get_maskfits('/data/users/meeg/minos3/cuts_new/', mrf)
    # tr.df.proc_convert()
    # tr.df.mfit_convert()

    tr.df.add_procs(prfs)
    tr.df.add_masks(mrfs)
    tr.df.proc_convert()
    tr.df.mfit_convert()

    hpixval = tr.df.maskFlags['badPixM']
    hcolval = tr.df.maskFlags['badColM']
    _ = tr.df.add_hotpixel(tr.df.mfs_full, hotpix[q], hpixval)
    _ = tr.df.add_hotcolmask(tr.df.mfs_full, hotcols[q], hcolval)
    ebase = enum
    tr.df.make_exp(520, 3200, ebase)
    tr.df.trim_all( 1, 513, 8, 3080)
    tr.df.trim_exp = tr.df.full_exp2d[1:513, 8:3080]

print("Proc files:", tr.df.proc_names, len(tr.df.procfits))
print("File shape:", tr.df.mfs[0].shape)


plen = len(pix_sizes)
flen = len(tr.df.procfits)
mlen = len(minos_halorad)

e_bins = [380, 22e3]
blen = len(e_bins)

reduced_pixels = np.zeros((mlen, plen, 2))
reduced_counts = np.zeros((mlen, plen, 4))
reduced_expos = np.zeros_like(reduced_pixels)


print('--------------------------------------------')
print(f"Doing fixed low radius with {lowfix}")
print('--------------------------------------------')

hval = minos_halorad[0].df.maskFlags['haloM']
mval = minos_halorad[0].df.maskFlags['edgeM']
for i,m in enumerate(minos_halorad):
    print("Working on quadrant", i)
#    _ = m.df.remove_maskval(m.df.mfs, mval) # Clear edge mask
#    _ = m.df.add_edgemask(m.df.mfs, mval, 60) # Insert new edge mask
    for j in range(len(m.df.sfs)):
        pixels, counts, expos = m.conv_cherenkov_low(m.df.sfs[j], m.df.mfs[j], [380, 22e3, -1], pix_sizes, lowrad=lowfix, include=True)
        reduced_pixels[i,:,:] += pixels[0]
        reduced_counts[i,:,:] += counts[0]
        reduced_expos[i,:,:] += expos[0]
        print(reduced_counts[i,:4,:])

print('------------------------')
print(pix_sizes)
print(reduced_counts)
print(reduced_expos)
print(reduced_pixels)
print('------------------------')
suffix = '_' + str(enum) + '_' + str(ccdnum)
sys.exit()
np.save(f'../data/minos3/{lowfix}/lowbin_expos_1e' + suffix, reduced_expos)
np.save(f'../data/minos3/{lowfix}/lowbin_count_1e' + suffix, reduced_counts)
np.save(f'../data/minos3/{lowfix}/lowbin_pixel_1e' + suffix, reduced_pixels)
