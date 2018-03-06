# scoubidou
Short cadence K2 light curves with PLD and celerite

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
