# scoubidou
Short cadence K2 light curves with PLD and celerite

Download the [data](https://archive.stsci.edu/missions/k2/target_pixel_files/c12/200100000/64000/ktwo200164267-c12_spd-targ.fits.gz) and save it as data/trappist1.fits.gz.

```
construct PLD basis:
  - All first order terms (~30)
  - PCA on second order terms (~30)
  - PCA on third order terms (~30)

mask outliers at X sigma:
  - easy

initial guess for GP (marg. over PLD):
  - quasi-periodic kernel (get Dan's help, see paper)

for i in niter:
  mask outliers at X sigma
  gradient descent to optimize GP
```
