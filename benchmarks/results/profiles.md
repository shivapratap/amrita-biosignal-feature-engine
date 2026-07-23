# ABFE performance profiles

## sample_entropy_profile_n256

checksum: 6621.966613024365

```text
         5262 function calls in 0.019 seconds

   Ordered by: cumulative time
   List reduced from 75 to 30 due to restriction <30>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.019    0.019 benchmark_baseline.py:227(checked_operation)
        1    0.000    0.000    0.019    0.019 benchmark_baseline.py:190(compute)
        1    0.000    0.000    0.019    0.019 sample_entropy_profile.py:94(sample_entropy_profile)
        1    0.000    0.000    0.019    0.019 sample_entropy_profile.py:79(_chm)
        2    0.002    0.001    0.015    0.007 sample_entropy_profile.py:68(_mean_cdf)
      508    0.000    0.000    0.011    0.000 fromnumeric.py:1446(searchsorted)
      510    0.000    0.000    0.011    0.000 fromnumeric.py:51(_wrapfunc)
      508    0.010    0.000    0.010    0.000 {method 'searchsorted' of 'numpy.ndarray' objects}
      511    0.005    0.000    0.005    0.000 {method 'sort' of 'numpy.ndarray' objects}
        3    0.000    0.000    0.003    0.001 _arraysetops_impl.py:145(unique)
        3    0.000    0.000    0.003    0.001 _arraysetops_impl.py:348(_unique1d)
      508    0.000    0.000    0.002    0.000 fromnumeric.py:968(sort)
        2    0.000    0.000    0.000    0.000 sample_entropy_profile.py:60(_distance_matrix)
        2    0.000    0.000    0.000    0.000 distance.py:2793(cdist)
        2    0.000    0.000    0.000    0.000 {built-in method scipy.spatial._distance_pybind.cdist_chebyshev}
      508    0.000    0.000    0.000    0.000 {method 'copy' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 _arraysetops_impl.py:1185(union1d)
        1    0.000    0.000    0.000    0.000 validation.py:13(validate_signal)
        1    0.000    0.000    0.000    0.000 fromnumeric.py:2589(all)
        1    0.000    0.000    0.000    0.000 fromnumeric.py:89(_wrapreduction_any_all)
        2    0.000    0.000    0.000    0.000 sample_entropy_profile.py:53(_templates)
        2    0.000    0.000    0.000    0.000 fromnumeric.py:3618(round)
        3    0.000    0.000    0.000    0.000 {method 'reduce' of 'numpy.ufunc' objects}
        2    0.000    0.000    0.000    0.000 {method 'round' of 'numpy.ndarray' objects}
        3    0.000    0.000    0.000    0.000 {method 'flatten' of 'numpy.ndarray' objects}
        2    0.000    0.000    0.000    0.000 _shape_base_impl.py:628(column_stack)
      513    0.000    0.000    0.000    0.000 {built-in method builtins.getattr}
      519    0.000    0.000    0.000    0.000 {built-in method numpy.asanyarray}
        1    0.000    0.000    0.000    0.000 _version.py:54(__init__)
      508    0.000    0.000    0.000    0.000 fromnumeric.py:964(_sort_dispatcher)
```

## sample_entropy_profile_n1024

checksum: 8096.8063394986666

```text
         20609 function calls in 0.160 seconds

   Ordered by: cumulative time
   List reduced from 66 to 30 due to restriction <30>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.160    0.160 benchmark_baseline.py:227(checked_operation)
        1    0.000    0.000    0.160    0.160 benchmark_baseline.py:190(compute)
        1    0.001    0.001    0.160    0.160 sample_entropy_profile.py:94(sample_entropy_profile)
        1    0.003    0.003    0.159    0.159 sample_entropy_profile.py:79(_chm)
        2    0.008    0.004    0.111    0.055 sample_entropy_profile.py:68(_mean_cdf)
     2047    0.072    0.000    0.072    0.000 {method 'sort' of 'numpy.ndarray' objects}
     2046    0.001    0.000    0.067    0.000 fromnumeric.py:51(_wrapfunc)
     2044    0.000    0.000    0.066    0.000 fromnumeric.py:1446(searchsorted)
     2044    0.065    0.000    0.065    0.000 {method 'searchsorted' of 'numpy.ndarray' objects}
        3    0.000    0.000    0.041    0.014 _arraysetops_impl.py:145(unique)
        3    0.002    0.001    0.041    0.014 _arraysetops_impl.py:348(_unique1d)
     2044    0.001    0.000    0.036    0.000 fromnumeric.py:968(sort)
        2    0.000    0.000    0.004    0.002 sample_entropy_profile.py:60(_distance_matrix)
        2    0.000    0.000    0.003    0.001 distance.py:2793(cdist)
        2    0.003    0.001    0.003    0.001 {built-in method scipy.spatial._distance_pybind.cdist_chebyshev}
     2044    0.001    0.000    0.001    0.000 {method 'copy' of 'numpy.ndarray' objects}
        3    0.001    0.000    0.001    0.000 {method 'flatten' of 'numpy.ndarray' objects}
        2    0.000    0.000    0.001    0.001 fromnumeric.py:3618(round)
        2    0.001    0.001    0.001    0.001 {method 'round' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 _arraysetops_impl.py:1185(union1d)
     2049    0.000    0.000    0.000    0.000 {built-in method builtins.getattr}
     2055    0.000    0.000    0.000    0.000 {built-in method numpy.asanyarray}
     2044    0.000    0.000    0.000    0.000 fromnumeric.py:964(_sort_dispatcher)
     2044    0.000    0.000    0.000    0.000 fromnumeric.py:1442(_searchsorted_dispatcher)
        2    0.000    0.000    0.000    0.000 fromnumeric.py:2338(sum)
        2    0.000    0.000    0.000    0.000 fromnumeric.py:69(_wrapreduction)
        3    0.000    0.000    0.000    0.000 {method 'reduce' of 'numpy.ufunc' objects}
        1    0.000    0.000    0.000    0.000 validation.py:13(validate_signal)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.empty}
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
```

## mixed_extractor_n1024

checksum: 1274.8526019227436

```text
         6637 function calls (6579 primitive calls) in 0.020 seconds

   Ordered by: cumulative time
   List reduced from 596 to 30 due to restriction <30>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.020    0.020 benchmark_baseline.py:227(checked_operation)
        1    0.000    0.000    0.020    0.020 benchmark_baseline.py:201(compute)
        1    0.000    0.000    0.020    0.020 extractor.py:426(extract)
        1    0.000    0.000    0.008    0.008 fuzzy.py:60(fuzzy_entropy)
        2    0.006    0.003    0.008    0.004 fuzzy.py:43(_similarity_mean)
        1    0.000    0.000    0.004    0.004 distribution.py:30(distribution_entropy)
        1    0.003    0.003    0.004    0.004 _histograms_impl.py:685(histogram)
        1    0.000    0.000    0.003    0.003 approximate.py:52(approximate_entropy)
        2    0.001    0.000    0.003    0.002 approximate.py:44(_phi)
        2    0.000    0.000    0.002    0.001 distance.py:2793(cdist)
        2    0.002    0.001    0.002    0.001 {built-in method scipy.spatial._distance_pybind.cdist_chebyshev}
        3    0.000    0.000    0.002    0.001 distance.py:1993(pdist)
        3    0.000    0.000    0.002    0.001 _lazy.py:57(lazy_apply)
        3    0.000    0.000    0.002    0.001 _lazy.py:333(wrapper)
        3    0.000    0.000    0.002    0.001 distance.py:2310(_np_pdist)
        3    0.002    0.001    0.002    0.001 {built-in method scipy.spatial._distance_pybind.pdist_chebyshev}
        2    0.000    0.000    0.001    0.000 _axis_nan_policy.py:419(axis_nan_policy_wrapper)
      153    0.001    0.000    0.001    0.000 {method 'reduce' of 'numpy.ufunc' objects}
        1    0.000    0.000    0.001    0.001 extractor.py:409(_compute_psd)
        1    0.000    0.000    0.001    0.001 psd.py:236(compute_psd)
        1    0.000    0.000    0.001    0.001 extractor.py:311(_package_version)
        1    0.000    0.000    0.001    0.001 __init__.py:980(version)
        1    0.000    0.000    0.001    0.001 _spectral_py.py:497(welch)
        1    0.000    0.000    0.001    0.001 _spectral_py.py:672(csd)
        1    0.000    0.000    0.001    0.001 time_domain.py:77(kurtosis)
        1    0.000    0.000    0.001    0.001 __init__.py:483(version)
        1    0.000    0.000    0.001    0.001 __init__.py:448(metadata)
        5    0.000    0.000    0.001    0.000 numeric.py:484(count_nonzero)
        4    0.000    0.000    0.000    0.000 {method 'sum' of 'numpy.ndarray' objects}
        4    0.000    0.000    0.000    0.000 _methods.py:49(_sum)
```

## batch_16x256

checksum: 2009.7313151506539

```text
         8660 function calls (8644 primitive calls) in 0.003 seconds

   Ordered by: cumulative time
   List reduced from 91 to 30 due to restriction <30>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.003    0.003 benchmark_baseline.py:227(checked_operation)
        1    0.000    0.000    0.003    0.003 benchmark_baseline.py:216(compute)
        1    0.000    0.000    0.003    0.003 extractor.py:540(extract_batch)
       16    0.000    0.000    0.003    0.000 extractor.py:426(extract)
      144    0.000    0.000    0.001    0.000 validation.py:13(validate_signal)
      144    0.000    0.000    0.000    0.000 fromnumeric.py:2589(all)
      288    0.000    0.000    0.000    0.000 numerictypes.py:475(issubdtype)
       16    0.000    0.000    0.000    0.000 time_domain.py:67(standard_deviation)
      256    0.000    0.000    0.000    0.000 {method 'reduce' of 'numpy.ufunc' objects}
       16    0.000    0.000    0.000    0.000 time_domain.py:133(zero_crossing_count)
      144    0.000    0.000    0.000    0.000 fromnumeric.py:89(_wrapreduction_any_all)
       16    0.000    0.000    0.000    0.000 time_domain.py:147(slope_sign_change_count)
       16    0.000    0.000    0.000    0.000 time_domain.py:43(maximum)
       17    0.000    0.000    0.000    0.000 extractor.py:320(_resolve_features)
       16    0.000    0.000    0.000    0.000 time_domain.py:57(mean)
       16    0.000    0.000    0.000    0.000 time_domain.py:127(waveform_length)
       16    0.000    0.000    0.000    0.000 time_domain.py:106(root_mean_square)
       16    0.000    0.000    0.000    0.000 fromnumeric.py:3869(std)
      128    0.000    0.000    0.000    0.000 warnings.py:170(simplefilter)
       16    0.000    0.000    0.000    0.000 time_domain.py:38(minimum)
       16    0.000    0.000    0.000    0.000 _methods.py:220(_std)
       99    0.000    0.000    0.000    0.000 {built-in method builtins.any}
      576    0.000    0.000    0.000    0.000 numerictypes.py:293(issubclass_)
       32    0.000    0.000    0.000    0.000 fromnumeric.py:3735(mean)
       16    0.000    0.000    0.000    0.000 _methods.py:150(_var)
       16    0.000    0.000    0.000    0.000 extractor.py:148(__post_init__)
      128    0.000    0.000    0.000    0.000 extractor.py:362(_value_diagnostic)
       32    0.000    0.000    0.000    0.000 _methods.py:117(_mean)
      128    0.000    0.000    0.000    0.000 warnings.py:188(_add_filter)
       48    0.000    0.000    0.000    0.000 fromnumeric.py:69(_wrapreduction)
```
