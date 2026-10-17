# cvtkpy

```
                   _   _
         _____   _| |_| | __
        / __\ \ / / __| |/ /
       | (__ \ V /| |_|   < 
        \___| \_/  \__|_|\_\
```

This is a Python package by Vince Buffalo that implements the methods of
Buffalo and Coop (2019, 2020) to calculate temporal covariances and convergence
correlations.

This repository, `cvtkpy-admixcov`, is a modified version of `cvtkpy` that
accounts for covariance components resulting from ongoing admixture.

The upstream `cvtkpy` package was split from http://github.com/vsbuffalo/cvtk,
which contains the original version of the code used in the analysis of Buffalo
and Coop (2020). See the [cvtk/notebooks](http://github.com/vsbuffalo/cvtk) for
examples from the 2020 PNAS paper on how to do these analyses; this is the most
detailed documentation/tutorial currently.

## Installation

Currently this Python package is not hosted anywhere. So you will need to
install it locally by cloning this repository,

```
$ git clone https://github.com/yuqingc7/cvtkpy-admixcov.git
```

Then installing,

```
$ python setup.py install  # must be Python 3
```

which should also install any dependencies you don't have.

Then, try:

```
$ python
Python 3.7.2 | packaged by conda-forge | (default, Mar 19 2019, 20:46:22)
[Clang 4.0.1 (tags/RELEASE_401/final)] :: Anaconda, Inc. on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import cvtk 
>>> # it works!
```

## Admixture covariance decomposition

Contributed by: Yuqing Chen

Codex and Claude were used to assist with code review and documentation.

This branch of `cvtkpy` adds `cvtk.admixture`, a small extension for temporal
samples whose ancestry proportions change through time. The implementation is
based on the model of Simon and Coop (2024), *The contribution of gene flow,
selection, and genetic drift to five thousand years of human allele frequency
change* ([PNAS, doi:10.1073/pnas.2312377121](https://doi.org/10.1073/pnas.2312377121)),
and the authors' MIT-licensed
[`py-admixcov` v1.0.0](https://github.com/alxsimon/py-admixcov/tree/v1.0.0)
reference implementation.

For observed allele frequencies `af`, ancestry proportions `Q`, and source
population allele frequencies `reference_af`, the implemented decomposition is

```text
predicted_admixture_af = Q @ reference_af

residual_covariance = observed_covariance
                      - admixture_covariance
                      - drift_admixture_error
```

### Observed temporal covariance

For locus `l`, the observed allele-frequency change in temporal interval `i`
is

```text
delta_p[i,l] = p[t+1,l] - p[t,l].
```

Before finite-sample correction, the raw entry for intervals `i` and `j` is
the centered population covariance across loci. For complete data,

```text
C_raw_uncorrected[i,j]
    = (1 / L) * sum_l((delta_p[i,l] - mean_l(delta_p[i,l]))
                      * (delta_p[j,l] - mean_l(delta_p[j,l]))).
```

The implementation calls `np.ma.cov(..., bias=True)`: changes in each interval
are centered across loci, non-finite values are masked, and the covariance uses
population normalization (`ddof=0`, rather than division by `L - 1`). CVTK then
applies its diploid finite-sample bias correction. The full matrix contains
every pair of temporal intervals, rather than only adjacent interval pairs.

### Ancestry-predicted covariance

For source population `k`, let `Q[t,k]` be its ancestry proportion at time `t`
and let `p_ref[k,l]` be its reference allele frequency at locus `l`. The allele
frequency predicted from ancestry composition alone is

```text
p_admix[t,l] = sum_k Q[t,k] * p_ref[k,l].
```

The predicted change is `delta_p_admix[i,l] = p_admix[t+1,l] - p_admix[t,l]`, and the direct admixture component is

```text
C_admix[i,j] = Cov_l(delta_p_admix[i,l], delta_p_admix[j,l]).
```

This term uses the same centered, masked `np.ma.cov(..., bias=True)` convention
as the uncorrected raw covariance, so subtracting it does not introduce an
uncentered mean-product term.

Because source allele frequencies are estimated from finite reference panels,
the implementation subtracts their expected sampling contribution. For each
source it calculates

```text
B[k] = mean_l(p_ref[k,l] * (1 - p_ref[k,l]) / (n_ref[k,l] - 1))
```

and subtracts `sum_k delta_Q[i,k] * B[k] * delta_Q[j,k]` from the corresponding
admixture-covariance entry.

In the LISS `Q_relative` analysis, the individual AQ ancestry estimates are
averaged within site and cohort. The two source proportions supplied to the
decomposition are

```text
Q[t] = (Q_HR[t], Q_AQ[t]), where Q_HR[t] = 1 - Q_AQ[t].
```

Thus, an increase or decrease in AQ ancestry contributes with its observed
sign to the direct ancestry-predicted allele-frequency changes.

### Drift-admixture covariance

Admixture can dilute a drift deviation inherited from an earlier interval.
Because this dilution produces a change that opposes the earlier deviation, it
can create negative off-diagonal temporal covariance even when new drift
increments are independent. To calculate this correction, the source whose
ancestry proportion increases in interval `i` is treated as the incoming
source. Its replacement fraction is

```text
alpha[i,k] = (Q[i+1,k] - Q[i,k]) / (1 - Q[i,k]).
```

All other source-specific alpha values in that interval are zero, and
`alpha_total[i]` is their sum. Interval drift variances are inferred by solving
the lower-triangular system implied by the diagonal of `C_raw - C_admix`. In
particular, the variance in interval `i` includes the new drift variance plus
earlier drift retained through intervening factors of `(1 - alpha_total)^2`.

The inferred drift variances are then propagated into each off-diagonal entry.
For `i < j`, the implementation sums the contribution of each drift increment
that occurred at or before interval `i`, multiplying it by the ancestry
replacement and retention factors connecting that increment to intervals `i`
and `j`. The resulting matrix is `C_drift_admix`; its diagonal is zero because
the diagonal information was used to infer the drift variances.

The final matrix is therefore

```text
C_residual = C_raw - C_admix - C_drift_admix.
```

The residual is covariance not explained by the fitted direct-admixture and
drift-admixture terms. It should not automatically be interpreted as selection.

### Model constraint and interpretation

The conversion from ancestry changes to `alpha` assumes that at most one
source increases in each interval. With the complementary two-source LISS
model this condition is satisfied mathematically: when AQ decreases, HR
increases and is treated as the incoming source, so the total interval alpha
is not silently set to zero. With more than two sources, the implementation
raises an error if multiple sources increase in the same interval.

The biological interpretation remains restrictive. Every change in the
two-source ancestry proportions is represented as a replacement pulse from the
source that increased. The model cannot distinguish actual immigration from
ancestry-biased survival, differential reproductive success, movement among
already admixed populations, or sampling error in `Q_relative`. The
drift-admixture term should therefore be interpreted as a correction under this
replacement model, not as direct evidence that a literal migration pulse
occurred in every interval.

The extension provides functions to:

- predict allele frequencies from ancestry proportions and reference
  frequencies;
- calculate observed and ancestry-predicted covariance matrices;
- convert changes in ancestry proportions into interval admixture fractions;
- estimate interval drift variances;
- construct the drift-admixture covariance correction;
- return raw, admixture, drift-error, and residual covariance components; and
- calculate full-span scalar `G`, `G_nc`, `G_nde`, and `Ap` summaries analogous
  to those used by Simon and Coop.

### Differences from `py-admixcov` v1.0.0

The ancestry-frequency prediction, conversion from ancestry changes to
admixture fractions, drift-variance solver, and drift-error equations were
adapted from `py-admixcov` v1.0.0, but the implementations are not numerically
identical. This version uses centered covariance with `ddof=0` rather than
`py-admixcov`'s `ddof=1`, and reports only full-span summary statistics using
the sum of the raw covariance matrix as the denominator. In contrast,
`py-admixcov` reports cumulative summaries and uses endpoint variance minus
pseudohaploid endpoint-bias terms in `get_summary` and its bootstrap.

Unlike the original application to pseudohaploid ancient-human data, observed
covariance here uses CVTK's diploid sampling correction. Reference-frequency
uncertainty is corrected using called allele counts from the source
populations. Unmodeled migration, inaccurate ancestry or reference
frequencies, demographic structure, and sampling variation can still
contribute to the residual.

## Citing

Please cite both Buffalo and Coop (*Genetics*, 2019) which is the original
paper about temporal covariance, and Buffalo and Coop (*PNAS*, 2020) which is
the correct paper to cite for these newer methods. Here are the BibTeX entries:

```
@ARTICLE{Buffalo2019-qs,
  title    = "The Linked Selection Signature of Rapid Adaptation in Temporal
              Genomic Data",
  author   = "Buffalo, Vince and Coop, Graham",
  journal  = "Genetics",
  volume   =  213,
  number   =  3,
  pages    = "1007--1045",
  month    =  nov,
  year     =  2019,
  language = "en"
}
@ARTICLE{Buffalo2020-my,
  title     = "Estimating the genome-wide contribution of selection to temporal
               allele frequency change",
  author    = "Buffalo, Vince and Coop, Graham",
  journal   = "Proc. Natl. Acad. Sci. U. S. A.",
  publisher = "National Academy of Sciences",
  month     =  aug,
  year      =  2020,
  language  = "en"
}
```

### Admixture covariance decomposition

If you use the `cvtk.admixture` covariance decomposition, please also cite:

```
@ARTICLE{Simon2024,
  title     = "The contribution of gene flow, selection, and genetic drift to
               five thousand years of human allele frequency change",
  author    = "Simon, Alexis and Coop, Graham",
  journal   = "Proc. Natl. Acad. Sci. U. S. A.",
  publisher = "National Academy of Sciences",
  volume    =  121,
  number    =  9,
  pages     = "e2312377121",
  month     =  feb,
  year      =  2024,
  doi       = "10.1073/pnas.2312377121",
  language  = "en"
}
```

Please also cite [this GitHub repository](https://github.com/yuqingc7/cvtkpy-admixcov).
