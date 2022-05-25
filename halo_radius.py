import AnalysisClass as ac
from AnalysisClass import Analysis
from AnalysisClass import DataFiles
import numpy as np
import sys


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
pix_sizes = [0,5,10,15, 60]
pix_sizes = [5]
# pix_sizes = np.arange(0,80,1)

ccdnum = int(sys.argv[1])
enum = int(sys.argv[2])

prfs = ['/data/users/meeg/minos3/proc_corr_proc_skp_moduleC40_41-ssc16_17-lta20_60_TEMP135K-run1_NROW520_NBINROW1_NCOL3200_NBINCOL1_EXPOSURE72000_CLEAR1800_1_250.fits']
#        '/data/users/meeg/minos3/proc_corr_proc_skp_moduleC40_41-ssc16_17-lta20_60_TEMP135K-run1_NROW520_NBINROW1_NCOL3200_NBINCOL1_EXPOSURE72000_CLEAR1800_1_255.fits',
#        '/data/users/meeg/minos3/proc_corr_proc_skp_moduleC40_41-ssc16_17-lta20_60_TEMP135K-run1_NROW520_NBINROW1_NCOL3200_NBINCOL1_EXPOSURE72000_CLEAR1800_1_260.fits',
#        '/data/users/meeg/minos3/proc_corr_proc_skp_moduleC40_41-ssc16_17-lta20_60_TEMP135K-run1_NROW520_NBINROW1_NCOL3200_NBINCOL1_EXPOSURE72000_CLEAR1800_1_265.fits']
        # '/data/users/meeg/minos3/proc_corr_proc_skp_moduleC40_41-ssc16_17-lta20_60_TEMP135K-run20_NROW520_NBINROW1_NCOL3200_NBINCOL1_EXPOSURE0_CLEAR1800_1_233.fits']


mrfs = ['/data/users/meeg/minos3/cuts_new/mask_corr_proc_skp_moduleC40_41-ssc16_17-lta20_60_TEMP135K-run1_NROW520_NBINROW1_NCOL3200_NBINCOL1_EXPOSURE72000_CLEAR1800_1_250.fits']
#          '/data/users/meeg/minos3/cuts_new/mask_corr_proc_skp_moduleC40_41-ssc16_17-lta20_60_TEMP135K-run1_NROW520_NBINROW1_NCOL3200_NBINCOL1_EXPOSURE72000_CLEAR1800_1_255.fits',
#          '/data/users/meeg/minos3/cuts_new/mask_corr_proc_skp_moduleC40_41-ssc16_17-lta20_60_TEMP135K-run1_NROW520_NBINROW1_NCOL3200_NBINCOL1_EXPOSURE72000_CLEAR1800_1_260.fits',
#          '/data/users/meeg/minos3/cuts_new/mask_corr_proc_skp_moduleC40_41-ssc16_17-lta20_60_TEMP135K-run1_NROW520_NBINROW1_NCOL3200_NBINCOL1_EXPOSURE72000_CLEAR1800_1_265.fits']

crfs = ['/data/users/meeg/minos3/cal_proc_corr_proc_skp_moduleC40_41-ssc16_17-lta20_60_TEMP135K-run1_NROW520_NBINROW1_NCOL3200_NBINCOL1_EXPOSURE72000_CLEAR1800_1_250.xml']

hotcols = np.load(f'../data/badcols_{ccdnum}.npy', allow_pickle=True)
hotpix = np.load(f'../data/badpix_{ccdnum}.npy', allow_pickle=True)

print("Running on ccd: %i and exposure: %i" % (ccdnum, enum))

for q in quads:
    cr = ac.Analysis(q,  1, badFlags)
    minos_halorad.append(cr)

bxs = [ 13, 220, 478, 490]
bys = [ 793, 2189, 1842, 2683]


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
#    tr.df.add_procs(prfs)
#    for ix,iy in zip(bxs, bys):
#        print("Pixels of interest:", tr.df.procfits[0][ix+1,iy+8])
#    tr.df.add_masks(mrfs)
#    tr.df.add_cal(crfs)
    tr.df.proc_convert()
#    tr.df.cal_convert()
    tr.df.mfit_convert()
    hpixval = tr.df.maskFlags['badPixM']
    hcolval = tr.df.maskFlags['badColM']
    _ = tr.df.add_hotpixel(tr.df.mfs_full, hotpix[q], hpixval)
    _ = tr.df.add_hotcolmask(tr.df.mfs_full, hotcols[q], hcolval)
    ebase = enum
    tr.df.make_exp(520, 3200, ebase)
    tr.df.trim_all( 1, 513, 8, 3080)
    tr.df.trim_exp = tr.df.full_exp2d[1:513, 8:3080]

print("MFS shape:", tr.df.mfs[0].shape)
print("SFS fits shape:", tr.df.sfs[0].shape)



plen = len(pix_sizes)
mlen = len(minos_halorad)

expos = np.zeros((mlen, plen, 2))
count = np.zeros((mlen, plen, 9))
pix = np.zeros((mlen, plen, 2))


hval = minos_halorad[0].df.maskFlags['haloM']
mval = minos_halorad[0].df.maskFlags['edgeM']
bval = minos_halorad[0].df.maskFlags['bleedM']




for i,m in enumerate(minos_halorad):
    print("Working on quadrant", i)
    print("First proc:", m.df.proc_names[:3])
    print("First masks:", m.df.mask_names[:3])
    for j,ps in enumerate(pix_sizes):
        print("Working on pix size:", ps)
        _ = m.df.remove_maskval(m.df.mfs, mval) # Clear edge mask
        _ = m.df.add_edgemask(m.df.mfs, mval, ps) # Insert new edge mask
        mspec = m.df.mfs[0][293,3011]
        _ = m.df.remove_maskval(m.df.mfs, hval) # Clear halo mask
        _ = m.df.conv_halomask(m.df.mfs, m.df.sfs, hval, ps)

        _ = m.df.remove_maskval(m.df.mfs, bval)
        _ = m.df.conv_bleedmask(m.df.mfs, m.df.sfs, bval, 100)
        sflags = m.df.masks_2_flags(m.df.mfs, badFlags)

        _ = m.df.remove_maskval(m.df.mfs, bval)
        _ = m.df.conv_bleedmask(m.df.mfs, m.df.sfs, bval, 50)
        sflags2 = m.df.masks_2_flags(m.df.mfs, badFlags_2e)

        exposure = [np.sum(m.df.partial_expo*sf) for sf in sflags]
        exposure_2 = [np.sum(m.df.partial_expo*sf) for sf in sflags2]
        expos_obj = (np.sum(exposure), np.sum(exposure_2))
        expos[i,j,:] += expos_obj

        pix1 = np.sum(sflags)
        pix2 = np.sum(sflags2)
        pixs_obj = (pix1, pix2)
        pix[i,j,:] += pixs_obj

        shoriz = np.array([[0,0,0],[1,1,1],[0,0,0]])
        svert = np.array([[0,1,0],[0,1,0],[0,1,0]])
        sdiag = np.array([[1,0,1],[0,1,0],[1,0,1]])
        ssolo = np.array([[0,0,0],[0,1,0],[0,0,0]])
        events = np.sum(m.one_pix_search(m.df.sfs, sflags, 1))

#        for ix,iy in zip(bxs, bys):
#            print("Pixels of interest:", m.df.sfs[0][ix,iy], m.df.mfs[0][ix,iy])

        mone = m.df.sfs[0] * sflags[0]
        gmone = np.where(mone==1)
        # print("ONE E:", [(x,y) for x,y in zip(*gmone)])
        # print(gmone)

        events_2 = np.sum(m.one_pix_search(m.df.sfs, sflags2, 2))
        _, one_clu = m.cluster_search(m.df.sfs, sflags, 1)
        kale, two_e = m.cluster_search(m.df.sfs, sflags2, 2)
        _, two_eh = m.cluster_search(m.df.sfs, sflags2, 2, shoriz)
        _, two_ev = m.cluster_search(m.df.sfs, sflags2, 2, svert)
        _, two_ed = m.cluster_search(m.df.sfs, sflags2, 2, sdiag)
        _, two_es = m.cluster_search(m.df.sfs, sflags2, 2, ssolo)
        _, three_e = m.cluster_search(m.df.sfs, sflags2, 3)
        events_obj = (events,events_2, two_e, two_eh, two_ev, two_ed, two_es, three_e, one_clu)
        count[i,j,:] += events_obj


print("Counts ", count)  


suffix = '_' + str(enum) + '_' + str(ccdnum)
# np.save('./data/minos3/trad_expos_1e' + suffix, expos)
# np.save('./data/minos3/trad_count_1e' + suffix, counts)
# np.save('./data/minos3/trad_pixel_1e' + suffix, pix)
