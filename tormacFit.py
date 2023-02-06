import numpy as np
import os
import pickle
import gzip
from scipy import interpolate
import matplotlib.pyplot as mpl
from time import time as TIME
import argparse
from emcee.autocorr import integrated_time

parser = argparse.ArgumentParser(description='Fit TORMAC models to data')
parser.add_argument('-a', action='store_true', \
                    help='use anisotropic models,(default=False)')
parser.add_argument('-n', default=100000, type=int, \
                    help='number of iterations (default = 100000)')
parser.add_argument('-b', default=10000, type=int, \
                    help='number of burn-in iterations (default=10000)')
parser.add_argument('-w', action='store_true', \
                    help='Do a WBIC run to estimate the evidence')

args = parser.parse_args()
anisotropic = args.a
niter = args.n
burn = args.b
wbic = args.w

if burn < niter:
    burn = niter // 10

then = TIME()


'''
Create a results directory if it doesn't exist yet
'''
path = './Results'
if not os.path.exists(path):
    os.makedirs(path)

'''
Load light curves
'''
timeJ, fluxJ, dfluxJ = np.loadtxt('J_corrected.txt', unpack=True)
timeH, fluxH, dfluxH = np.loadtxt('H_corrected.txt', unpack=True)
timeK, fluxK, dfluxK = np.loadtxt('K_corrected.txt', unpack=True)

beta_ = 1.0
if wbic:
    n_ = len(timeJ) + len(timeH) + len(timeK)
    beta_ = 1.0 / np.log(n_)

'''
Normalize the fluxes to something less teeny
'''
fluxJ /= 1.e-15
dfluxJ /= 1.e-15

fluxH /= 1.e-15
dfluxH /= 1.e-15

fluxK /= 1.e-15
dfluxK /= 1.e-15


'''
Load model grids
'''
sfx = 'aniOff'
if anisotropic:
    sfx = 'aniOn'

fp = gzip.open('tormac_J_' + sfx + '.pkl.gz', 'r')
jdb = pickle.load(fp, encoding='latin1')
fp.close()

fp = gzip.open('tormac_H_' + sfx + '.pkl.gz', 'r')
hdb = pickle.load(fp, encoding='latin1')
fp.close()

fp = gzip.open('tormac_K_' + sfx + '.pkl.gz', 'r')
kdb = pickle.load(fp, encoding='latin1')
fp.close()

jGrid = jdb['respJ']
hGrid = hdb['respH']
kGrid = kdb['respK']

'''
Edit the data
We need to be able to interpolate the model grid onto the observed grid.
If the times in the data exceed the times in the model grid, it won't work.
Toss out data with times > model grid.
'''
idx = timeJ <= max(jdb['times'])
timeJ = timeJ[idx]
fluxJ = fluxJ[idx]
dfluxJ = dfluxJ[idx]


idx = timeH <= max(hdb['times'])
timeH = timeH[idx]
fluxH = fluxH[idx]
dfluxH = dfluxH[idx]


idx = timeK <= max(kdb['times'])
timeK = timeK[idx]
fluxK = fluxK[idx]
dfluxK = dfluxK[idx]

'''
For convenience, fold the data into a dictionary
'''
lightCurves = dict()
lightCurves['timeJ'] = timeJ
lightCurves['timeH'] = timeH
lightCurves['timeK'] = timeK
lightCurves['fluxJ'] = fluxJ
lightCurves['fluxH'] = fluxH
lightCurves['fluxK'] = fluxK
lightCurves['dfluxJ'] = dfluxJ
lightCurves['dfluxH'] = dfluxH
lightCurves['dfluxK'] = dfluxK


'''
We need to define the parameters.
It looks like the models need to be scaled to the data, so we have
0 - sigma (10 to 45)
1 - Y (10 to 500)
2 - q (-2 to 2)
3 - incl (0 to 90)
4 - numclouds (set to 50000) - this is fixed for now, and so we'll not use it 

Potentially, "scale" can be a dependent parameter, but for now we'll keep it free.
Could also use a "sysunc" parameter. Save it for later.
'''
jGrid = jGrid[:,:,:,:,:,0] # get rid of numclouds
hGrid = hGrid[:,:,:,:,:,0] # get rid of numclouds
kGrid = kGrid[:,:,:,:,:,0] # get rid of numclouds

# set up a hypergrid interpolator
fitPoints = [jdb['times'], jdb['sigma'], jdb['Y'], jdb['q'], jdb['incl']]
interpJ = interpolate.RegularGridInterpolator(fitPoints, jGrid)

fitPoints = [hdb['times'], hdb['sigma'], hdb['Y'], hdb['q'], hdb['incl']]
interpH = interpolate.RegularGridInterpolator(fitPoints, hGrid)

fitPoints = [kdb['times'], kdb['sigma'], kdb['Y'], kdb['q'], kdb['incl']]
interpK = interpolate.RegularGridInterpolator(fitPoints, kGrid)

# bundle interpolants into a dictionary
models = dict()
models['interpJ'] = interpJ
models['interpH'] = interpH
models['interpK'] = interpK

def responseModel(p, times=timeH, interpfn=interpH):
    # to get the time series, we need to set up an array of values
    # to evaluate the interpolation function
    _null = times * 0
    sigma = p[0] + _null
    Y = p[1] + _null
    q = p[2] + _null
    incl = p[3] + _null
    # print('p = ', p)
    evalPoints = np.array([times, sigma, Y, q, incl])
    #model = p[0] * interpfn(evalPoints.T)
    model = interpfn(evalPoints.T)
    return model

def calculateScale(m, y, dy):
    # treat the scale as determinative
    scl = (y*m/dy**2).sum() / ((m/dy)**2).sum()
    #print('scl = ', scl)
    return scl

def calculateModel(p, lightCurves=lightCurves, models=models):
    modelJ = responseModel(p, lightCurves['timeJ'], interpfn=models['interpJ'])
    modelH = responseModel(p, lightCurves['timeH'], interpfn=models['interpH'])
    modelK = responseModel(p, lightCurves['timeK'], interpfn=models['interpK'])
    return (modelJ, modelH, modelK)

def loglikelihood(p, lightCurves=lightCurves, models=models):
    modelJ, modelH, modelK = calculateModel(p, lightCurves, models)
    model = np.concatenate((modelJ, modelH, modelK))
    #print('model = ', model)
    data = np.concatenate((lightCurves['fluxJ'], lightCurves['fluxH'], \
                           lightCurves['fluxK']))
    '''
    Add in systemic uncertainties by quadrature
    '''
    dJ = np.sqrt(lightCurves['dfluxJ']**2 + (p[4]*lightCurves['fluxJ']/100)**2)
    dH = np.sqrt(lightCurves['dfluxH']**2 + (p[5]*lightCurves['fluxH']/100)**2)
    dK = np.sqrt(lightCurves['dfluxK']**2 + (p[6]*lightCurves['fluxK']/100)**2)
    err = np.concatenate((dJ, dH, dK))
    
    scl = calculateScale(model, data, err)
    zscore = (data - scl*model) / err
    logP_ = -0.5 * (np.log(2*np.pi*err**2)  + zscore**2)
    return logP_.sum()

'''
Try a pydream implementation
Note: jackdream is a slightly modified version of pydream that changes thinning behavior
and also allows the use of a thermodynamic inverse temperature, beta.
'''

from jackdream.parameters import FlatParam, SampledParam
from scipy.stats import norm, uniform
from jackdream.core import run_dream
from jackdream.convergence import Gelman_Rubin

sigma = SampledParam(uniform, loc=min(hdb['sigma']), scale=max(hdb['sigma'])-min(hdb['sigma']))
Y = SampledParam(uniform, loc=min(hdb['Y']), scale=max(hdb['Y'])-min(hdb['Y']))
q = SampledParam(uniform, loc=min(hdb['q']), scale=max(hdb['q'])-min(hdb['q']))
incl = SampledParam(uniform, loc=min(hdb['incl']), scale=max(hdb['incl'])-min(hdb['incl']))
'''
Add (fractional) systematic uncertainties in each band
For now, cap at 100%
'''
sysJ = SampledParam(uniform, loc=0, scale=100.0)
sysH = SampledParam(uniform, loc=0, scale=100.0)
sysK = SampledParam(uniform, loc=0, scale=100.0)

sampledParameters = [sigma, Y, q, incl, sysJ, sysH, sysK]

#burn = 10000 # for now, just assume the burn-in is 10000
sampled_params, log_ps = run_dream(sampledParameters, loglikelihood, beta=beta_,\
                                   niterations=niter, nchains=3, history_thin=1, \
                                   start_random=True, verbose=False)
Rhat = np.amax(Gelman_Rubin(sampled_params)) # convergence statistic


'''
Turn the sampled parameters list into an array
'''
print('shape of sampled params = ', np.array(sampled_params).shape)
sampled_params = np.array(sampled_params)[:,burn:,:]
log_ps = np.array(log_ps).squeeze()[:,burn:]

'''
Fold the chains together
'''
print('sampled_params = ', sampled_params)
print('sampled_params[:,:,0]: ', sampled_params[:,:,0])
sigmaSamples = sampled_params[:,:,0].flatten(order='F')
print('sigmaSamples = ', sigmaSamples)
YSamples = sampled_params[:,:,1].flatten(order='F')
qSamples = sampled_params[:,:,2].flatten(order='F')
inclSamples = sampled_params[:,:,3].flatten(order='F')
sysJSamples = sampled_params[:,:,4].flatten(order='F')
sysHSamples = sampled_params[:,:,5].flatten(order='F')
sysKSamples = sampled_params[:,:,6].flatten(order='F')
log_ps = log_ps.flatten(order='F')

'''
Need to recalculate scales, which, being determinative, aren't stored by jackdream.
It's also a good opportunity to create model stacks. For now, though, just calculate mean
and std of models
'''
from tqdm import tqdm
print('Re-calculating scales')
data = np.concatenate((fluxJ, fluxH, fluxK))
err = np.concatenate((dfluxJ, dfluxH, dfluxK))
sclSamples = 0.0 * sigmaSamples

modelsum = 0.0 * data
modelsqrsum  = 0.0 * data

for i in tqdm(range(len(sigmaSamples))):
    p = np.array((sigmaSamples[i], \
                  YSamples[i], \
                  qSamples[i], \
                  inclSamples[i], \
                  sysJSamples[i], \
                  sysHSamples[i], \
                  sysKSamples[i]))
    modelJ, modelH, modelK = calculateModel(p, models=models)
    model = np.concatenate((modelJ, modelH, modelK))
    scl = calculateScale(model, data, err)
    sclSamples[i] = scl
    modelsum += scl*model
    modelsqrsum += (scl*model)**2

modelmean = modelsum / len(sigmaSamples)
model2mean = modelsqrsum / len(sigmaSamples)
modelstd = np.sqrt(model2mean - modelmean**2)


'''
Save the results to a pandas/hdf5 file
'''

if wbic:
    sfx = sfx + '-wbic'

import pandas as pd

results = {}
results['sigma'] = sigmaSamples
results['Y'] = YSamples
results['q'] = qSamples
results['incl'] = inclSamples
results['sysJ'] = sysJSamples
results['sysH'] = sysHSamples
results['sysK'] = sysKSamples
results['log_ps'] = log_ps
results['scl'] = sclSamples

df = pd.DataFrame(data=results)
df.to_hdf('parameters-' + sfx + '.hdf5', key='df', mode='w')

results = {}
results['timeJ'] = timeJ
results['modelJmean'] = modelmean[0:len(timeJ)]
results['modelJstd'] = modelstd[0:len(timeJ)]
results['timeH'] = timeH
results['modelHmean'] = modelmean[len(timeJ):len(timeJ)+len(timeH)]
results['modelHstd'] = modelstd[len(timeJ):len(timeJ)+len(timeH)]
results['timeK'] = timeK
results['modelKmean'] = modelmean[len(timeJ)+len(timeH):]
results['modelKstd'] = modelstd[len(timeJ)+len(timeH):]

df.to_hdf('models-anioff.hdf5', key='df', mode='w')

'''
calculate number of independent samples = length / iat
'''
N = len(sigmaSamples)
nSigma = N // integrated_time(sigmaSamples, quiet=True)
nY = N // integrated_time(YSamples, quiet=True)
nq = N // integrated_time(qSamples, quiet=True)
nIncl = N // integrated_time(inclSamples, quiet=True)
nSysJ = N // integrated_time(sysJSamples, quiet=True)
nSysH = N // integrated_time(sysHSamples, quiet=True)
nSysK = N // integrated_time(sysKSamples, quiet=True)


'''
Create a summary report
'''
from sigfig import round as sround
outfile = './Results/summaryReport-' + sfx + '.txt'
fp = open(outfile, 'w')
print('sigma: ', sround(sigmaSamples.mean(), sigmaSamples.std()), nSigma, file=fp)
print('Y: ', sround(YSamples.mean(), YSamples.std()), nY, file=fp)
print('q: ', sround(qSamples.mean(), qSamples.std()), nq, file=fp)
print('incl: ', sround(inclSamples.mean(), inclSamples.std()), nIncl, file=fp)
print('sysJ: ', sround(sysJSamples.mean(), sysJSamples.std()), nSysJ, file=fp)
print('sysH: ', sround(sysHSamples.mean(), sysHSamples.std()), nSysH, file=fp)
print('sysK: ', sround(sysKSamples.mean(), sysKSamples.std()), nSysK, file=fp)
print('scl: ', sround(sclSamples.mean(), sclSamples.std()), file=fp)
print('log_ps: ', sround(log_ps.mean(), log_ps.std()), file=fp)
print('beta: ', beta_, file=fp)
print('Rhat: ', Rhat, file=fp)
fp.close()

'''
Plot histograms of the main parameters
'''
f1 = mpl.figure(1, figsize=(8,6))
f1.clf()
ax1 = f1.add_subplot(221)
mpl.hist(sigmaSamples, bins=30, density=True, \
         edgecolor='k', color='gray')
ax1.set_xlabel('$\\sigma$ (degrees)')
ax1.set_ylabel('Posterior Probability Density')
mpl.tight_layout()

ax2 = f1.add_subplot(222)
mpl.hist(YSamples, bins=30, density=True, \
         edgecolor='k', color='gray')
ax2.set_xlabel('$Y$')
mpl.tight_layout()

ax3 = f1.add_subplot(223)
mpl.hist(qSamples, bins=30, density=True, \
         edgecolor='k', color='gray')
ax3.set_xlabel('$q$')
ax3.set_ylabel('Posterior Probability Density')
mpl.tight_layout()


ax4 = f1.add_subplot(224)
mpl.hist(inclSamples, bins=30, density=True, \
         edgecolor='k', color='gray')
ax4.set_xlabel('$i$ (degrees)')
f1.tight_layout()

f1.savefig('./Results/parameterHistograms-' + sfx + '.pdf')

'''
Plot models on data
'''

'''
Plot models on data
'''

def plotTimeSeries(ax, t, flux, dflux, model, dmodel):
    ax.errorbar(t, flux, dflux, fmt='k.')
    '''
    find gaps in the time axis
    for now, break the data up when the gap is greater than 2 days
    '''
    idx = np.arange(len(t), dtype=int) # set up an array of indices
    dt = np.diff(t) # calculate time differences
    idx = idx[1:][dt > 2] # identify where gaps are > 2 days

    '''
    Need to concatenate start and end indices
    '''
    idx = np.insert(idx,0,0)
    idx = np.append(idx,len(t)-1)

    # loop over gapped data
    for j in range(1,len(idx)):
        i = idx[j]
        i_ = idx[j-1]
        t_ = t[i_:i]
        m_ = model[i_:i]
        dm_ = dmodel[i_:i]
        ax.fill_between(t_, m_-dm_, m_+dm_, color='r', alpha=0.3)
        ax.plot(t_, m_, 'r-')
    return
    

f2 = mpl.figure(2)
f2.clf()
ax1 = f2.add_subplot(311)
t = timeJ
m = modelmean[0:len(timeJ)]
dm = modelstd[0:len(timeJ)]
plotTimeSeries(ax1, t, fluxJ, dfluxJ, m, dm)
ax1.text(0.1, 0.8, 'J', transform=ax1.transAxes)
ax1.xaxis.set_ticklabels([])
    
ax2 = f2.add_subplot(312)
t = timeH
m = modelmean[len(timeJ):len(timeJ)+len(timeH)]
dm = modelstd[len(timeJ):len(timeJ)+len(timeH)]
plotTimeSeries(ax2, t, fluxH, dfluxH, m, dm)
ax2.set_ylabel('Flux Density')
ax2.text(0.1, 0.8, 'H', transform=ax2.transAxes)
ax2.xaxis.set_ticklabels([])

ax3 = f2.add_subplot(313)
t = timeK
m = modelmean[len(timeJ)+len(timeH):]
dm = modelstd[len(timeJ)+len(timeH):]
plotTimeSeries(ax3, t, fluxK, dfluxK, m, dm)
ax3.set_xlabel('Time (days)')
ax3.text(0.1, 0.8, 'K', transform=ax3.transAxes)

f2.tight_layout()
f2.savefig('./Results/lightCurve-fits-' + sfx + '.pdf')

'''
Histograms of the systematic uncertainties
'''
f3 = mpl.figure(3, figsize=(8,3))
f3.clf()
ax1 = f3.add_subplot(131)
ax1.hist(sysJSamples, bins=30, color='gray', edgecolor='k', density=True)
ax1.text(0.1, 0.8, 'J', transform=ax1.transAxes)
ax1.set_ylabel('Posterior Probability Density')

ax2 = f3.add_subplot(132)
ax2.hist(sysHSamples, bins=30, color='gray', edgecolor='k', density=True)
ax2.text(0.1, 0.8, 'H', transform=ax2.transAxes)
ax2.set_xlabel('Systematic Uncertainty (%)')

ax3 = f3.add_subplot(133)
ax3.hist(sysKSamples, bins=30, color='gray', edgecolor='k', density=True)
ax3.text(0.1, 0.8, 'K', transform=ax3.transAxes)

f3.tight_layout()
f3.savefig('./Results/sysUncertainties-' + sfx + '.pdf')

now = TIME()
elapsed = (now - then) / 60.
print('Run time = ', elapsed, ' minutes')

if wbic:
    print('This was a WBIC run. An estimate for the log evidence is given by')
    print('the value of log_ps in Results/' + outfile + '.')
          
