"""Load the TRAPPIST-1 short cadence data and do the initial de-trending."""
try:
    import pyfits
except ImportError:
    import astropy.io.fits as pyfits
import numpy as np
from scipy.signal import savgol_filter
import os


def initial_flux(clobber=False):
    """Return an initial guess at the de-trended light curve."""
    if (not clobber) and os.path.exists("data/initial.npz"):
        data = np.load("data/initial.npz")
        time = data['time']
        fpix = data['fpix']
        detrended_flux = data['detrended_flux']
        outliers = data['outliers']
        transits = data['transits']
        return time, fpix, detrended_flux, outliers, transits

    # Load the target pixel file
    with pyfits.open('data/trappist1.fits.gz') as file:
        time = file[1].data.field('TIME')
        fpix = file[1].data.field('FLUX')
        fpix_err = file[1].data.field('FLUX_ERR')
        quality = file[1].data.field('QUALITY')
        naninds = np.where(np.isnan(time)[0])
        time = np.delete(time, naninds)
        fpix = np.delete(fpix, naninds, axis=0)
        fpix_err = np.delete(fpix_err, naninds, axis=0)
        quality = np.delete(quality, naninds)

    # Get a generous aperture
    aperture = np.zeros(fpix.shape[1:], dtype=int)
    aperture[2:9, 3:9] = 1
    ap = np.where(aperture & 1)
    fpix = np.array([f[ap] for f in fpix], dtype='float64')

    # Compute the SAP flux
    flux = np.sum(fpix, axis=1)

    # Get bad cadences
    bad_bits = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17]
    badmask = []
    for b in bad_bits:
        badmask += list(np.where(quality & 2 ** (b - 1))[0])
    badmask = np.array(badmask, dtype=int)

    # Get nan fluxes
    nanflux = np.where(np.isnan(flux))[0]

    # Join
    badmask = np.array(list(set(badmask).union(set(nanflux))), dtype=int)

    # Delete from light curve for simplicity
    time = np.delete(time, badmask)
    fpix = np.delete(fpix, badmask, axis=0)
    fpix_err = np.delete(fpix_err, badmask, axis=0)
    quality = np.delete(quality, badmask)
    flux = np.sum(fpix, axis=1)

    # Normalize the light curve
    baseline = np.nanmean(flux)
    fpix /= baseline
    fpix_err /= baseline

    # Recompute the total flux
    flux = np.sum(fpix, axis=1)

    # Initial sigma clipping (aggressive at 3 sigma)
    med = np.nanmedian(flux)
    MAD = 1.4826 * np.nanmedian(np.abs(flux - med))
    outliers = np.where((flux > med + 3. * MAD) | (flux < med - 3. * MAD))[0]

    # Initial first order PLD with a 50th order polynomial mean model
    A = fpix.reshape(len(time), -1) / flux.reshape(-1, 1)
    A = np.hstack((A, np.array([np.linspace(0, 1, len(time)) ** n
                                for n in range(50)]).T))

    # Mask the outliers
    A_ = np.delete(A, outliers, axis=0)

    # Solve the linear equation
    w = np.linalg.solve(np.dot(A_.T, A_), np.dot(
        A_.T, np.delete(flux, outliers)))
    model = np.dot(A, w)
    detrended_flux = 1 + flux - model

    # Do sigma-clipping again
    smooth_flux = np.interp(time, np.delete(time, outliers),
                            np.delete(detrended_flux, outliers))
    smooth_flux = savgol_filter(smooth_flux, 1501, 2)

    whitened_flux = detrended_flux - smooth_flux
    med = np.nanmedian(whitened_flux)
    MAD = 1.4826 * np.nanmedian(np.abs(whitened_flux - med))
    outliers = np.where((whitened_flux > med + 3. * MAD) |
                        (whitened_flux < med - 3. * MAD))[0]

    # Sweet, now run our PLD model once more for a
    # better intial guess at the light curve
    A_ = np.delete(A, outliers, axis=0)

    # Solve the linear equation
    w = np.linalg.solve(np.dot(A_.T, A_), np.dot(
        A_.T, np.delete(flux, outliers)))
    model = np.dot(A, w)
    detrended_flux = 1 + flux - model

    # Get the transit times
    dur = 0.025
    transits = []
    for planet in ['b', 'c', 'd', 'e', 'f', 'g', 'h']:
        ttimes, _ = np.loadtxt("data/" + planet + ".ttv", unpack=True)
        ttimes -= 2454833
        ttimes = ttimes[(ttimes > time[0]) & (ttimes < time[-1])]
        for t in ttimes:
            transits.extend(np.where(np.abs(time - t) < dur)[0])
    transits = np.array(list(sorted(list(set(transits)))), dtype=int)

    # Save
    data = np.savez("data/initial.npz", time=time, fpix=fpix,
                    detrended_flux=detrended_flux,
                    outliers=outliers, transits=transits)

    # Return
    return time, fpix, detrended_flux, outliers, transits


if __name__ == "__main__":

    initial_flux(clobber=True)
