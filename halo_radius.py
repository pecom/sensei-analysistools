import AnalysisClass as ac
from AnalysisClass import Analysis
from AnalysisClass import DataFiles
import numpy as np
import sys


badFlags = { 'badColM', 'badPixM', 'bleedM', 'crosstalkM', 'edgeM',
             'extendBleedM', 'fullWellM', 'haloM', 'looseCluM', 
             'lowECluM', 'neighborM', 'noisyM', 'serialM'
}

badFlags_2e = { 'badColM', 'badPixM', 'bleedM', 'crosstalkM', 'edgeM',
             'extendBleedM', 'fullWellM', 'haloM', 'looseCluM', 
             'lowECluM', 'noisyM', 'serialM'
}

quads = [0,1,2,3]
# quads = [3]
minos_halorad = []
# pix_sizes = np.arange(10, 110, 2)
pix_sizes = np.arange(0,40,1)

ccdnum = int(sys.argv[1])
enum = int(sys.argv[2])


hotcols = np.load(f'./data/badcols_{ccdnum}.npy', allow_pickle=True)
hotpix = np.load(f'./data/badpix_{ccdnum}.npy', allow_pickle=True)

print("Running on ccd: %i and exposure: %i" % (ccdnum, enum))

for q in quads:
    cr = ac.Analysis(q,  1, badFlags)
    minos_halorad.append(cr)

for q, tr in zip(quads, minos_halorad):
        #     tr.df.get_samplefiles('../AnalysisTools/DanielAnalysis/minos3data/', "valuesE_1.*quad"+str(q))€
    print("On quadrant: ", q)
    newBp = {
        "zeroOneCut": .7,#51,#51,#.63, #0.95 #cutoff (in pixels) between 0e and 1e bin
        "oneTwoCut": 1.63, #1.8 #cutoff (in pixels) between 1e and 2e bin
        "twoThreeCut": 2.63, #cutoff (in pixels) between 2e and 3e bin
        "bottomCut": -1.2, #minimum value that we will consider a "valid" 0-e event
        "countCut": 0.85, #Cut to use for statistical conversion (0.85 was found to be optimal)
        "fractional2s" : True #Shoudl we calcualte the number of 2s passing the cut, or add the probability?
    }
    tr.df.set_bp(newBp)
    tr.df.set_minos3()
    prf = 'proc_corr_proc_skp.*EXPOSURE%i.*_%i_.*.fits' % (enum, ccdnum)
    mrf = 'mask_corr_proc_skp.*EXPOSURE%i.*_%i_.*.fits' % (enum, ccdnum)
    tr.df.get_fits('/data/users/meeg/minos3/', prf)
    tr.df.get_maskfits('/data/users/meeg/minos3/cuts_new/', mrf)
    tr.df.proc_convert()
    tr.df.mfit_convert()
    ebase = enum
    tr.df.make_exp(520, 3200, ebase)
    tr.df.trim_all( 1, 513, 8, 3080)
    tr.df.trim_exp = tr.df.full_exp2d[1:513, 8:3080]

print("Mask fits shape:", tr.df.maskfits[0].shape)
print("Proc fits shape:", tr.df.procfits[0].shape)



plen = len(pix_sizes)
mlen = len(minos_halorad)

exposure = np.zeros((mlen, plen), dtype=float)
one_count = np.zeros((mlen, plen), dtype=int)
pix_count = np.zeros_like(one_count)
two_1p_count = np.zeros_like(one_count)
two_2p_count = np.zeros_like(one_count)
exposure2 = np.zeros((mlen, plen), dtype=float)
pix_count2 = np.zeros_like(one_count)

hval = minos_halorad[0].df.maskFlags['haloM']
mval = minos_halorad[0].df.maskFlags['edgeM']
hpixval = minos_halorad[0].df.maskFlags['badPixM']
hcolval = minos_halorad[0].df.maskFlags['badColM']

# apply hot column and hot pixel masks:
for q, m in zip(quads, minos_halorad):
    _ = m.df.add_hotpixel(m.df.mfs, hotpix[q], hpixval)
    _ = m.df.add_hotcolmask(m.df.mfs, hotcols[q], hcolval)
    print("Applied hot col and hot pix masks")


for i,m in enumerate(minos_halorad):
    print("Working on quadrant", i)
    print("First proc:", m.df.proc_names[:3])
    print("First masks:", m.df.mask_names[:3])
    for j,ps in enumerate(pix_sizes):
        print("Working on pix size:", ps)
        _ = m.df.remove_maskval(m.df.mfs, mval) # Clear edge mask
        _ = m.df.add_edgemask(m.df.mfs, mval, ps) # Insert new edge mask
        _ = m.df.remove_maskval(m.df.mfs, hval) # Clear halo mask
        # _ = m.df.add_halomask(m.df.mfs, m.df.sfs, hval, ps) # Insert new halo mask
        _ = m.df.conv_halomask(m.df.mfs, m.df.sfs, hval, ps)
        sflags = m.df.masks_2_flags(m.df.mfs, badFlags)
        sflags2 = m.df.masks_2_flags(m.df.mfs, badFlags_2e)

        pix_count[i,j] = np.sum(sflags)
        exposure[i,j] = np.sum(m.df.trim_exp * sflags)

        pix_count2[i,j] = np.sum(sflags2)
        exposure2[i,j] = np.sum(m.df.trim_exp * sflags2)

        one_results = m.one_pix_search(m.df.sfs, sflags, 1)
        two_results = m.one_pix_search(m.df.sfs, sflags2, 2)
        kale, two2p_results = m.cluster_search(m.df.sfs, sflags2, 2)
        print("One counts:", two_results)
        one_count[i,j] = np.sum(one_results)
        two_1p_count[i,j] = np.sum(two_results)
        two_2p_count[i,j] = two2p_results

suffix = '_' + str(enum) + '_' + str(ccdnum)
np.save('./data/minos3/mod_expos_1e' + suffix, exposure)
np.save('./data/minos3/mod_count_1e' + suffix, one_count)
np.save('./data/minos3/mod_pixel_1e' + suffix, pix_count)
np.save('./data/minos3/mod_count_1pix2e' + suffix, two_1p_count)
np.save('./data/minos3/mod_count_2pix2e' + suffix, two_2p_count)
np.save('./data/minos3/mod_expos_2e' + suffix, exposure2)
np.save('./data/minos3/mod_pixel_2e' + suffix, pix_count2)
