# ABFE performance profiles

## sample_entropy_profile_n256

checksum: 6621.966613024365

```text
         5262 function calls in 0.021 seconds

   Ordered by: cumulative time
   List reduced from 75 to 30 due to restriction <30>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.021    0.021 benchmark_baseline.py:227(checked_operation)
        1    0.000    0.000    0.021    0.021 benchmark_baseline.py:190(compute)
        1    0.000    0.000    0.021    0.021 sample_entropy_profile.py:94(sample_entropy_profile)
        1    0.000    0.000    0.020    0.020 sample_entropy_profile.py:79(_chm)
        2    0.002    0.001    0.017    0.008 sample_entropy_profile.py:68(_mean_cdf)
      508    0.000    0.000    0.012    0.000 fromnumeric.py:1446(searchsorted)
      510    0.000    0.000    0.012    0.000 fromnumeric.py:51(_wrapfunc)
      508    0.011    0.000    0.011    0.000 {method 'searchsorted' of 'numpy.ndarray' objects}
      511    0.005    0.000    0.005    0.000 {method 'sort' of 'numpy.ndarray' objects}
        3    0.000    0.000    0.003    0.001 _arraysetops_impl.py:145(unique)
        3    0.000    0.000    0.003    0.001 _arraysetops_impl.py:348(_unique1d)
      508    0.000    0.000    0.003    0.000 fromnumeric.py:968(sort)
        2    0.000    0.000    0.000    0.000 sample_entropy_profile.py:60(_distance_matrix)
        2    0.000    0.000    0.000    0.000 distance.py:2793(cdist)
      508    0.000    0.000    0.000    0.000 {method 'copy' of 'numpy.ndarray' objects}
        2    0.000    0.000    0.000    0.000 {built-in method scipy.spatial._distance_pybind.cdist_chebyshev}
        1    0.000    0.000    0.000    0.000 _arraysetops_impl.py:1185(union1d)
        1    0.000    0.000    0.000    0.000 validation.py:13(validate_signal)
        2    0.000    0.000    0.000    0.000 fromnumeric.py:3618(round)
        2    0.000    0.000    0.000    0.000 {method 'round' of 'numpy.ndarray' objects}
        2    0.000    0.000    0.000    0.000 sample_entropy_profile.py:53(_templates)
        3    0.000    0.000    0.000    0.000 {method 'reduce' of 'numpy.ufunc' objects}
        3    0.000    0.000    0.000    0.000 {method 'flatten' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 fromnumeric.py:2589(all)
      513    0.000    0.000    0.000    0.000 {built-in method builtins.getattr}
        2    0.000    0.000    0.000    0.000 fromnumeric.py:2338(sum)
      519    0.000    0.000    0.000    0.000 {built-in method numpy.asanyarray}
        1    0.000    0.000    0.000    0.000 fromnumeric.py:89(_wrapreduction_any_all)
        2    0.000    0.000    0.000    0.000 fromnumeric.py:69(_wrapreduction)
      508    0.000    0.000    0.000    0.000 fromnumeric.py:964(_sort_dispatcher)
```

## sample_entropy_profile_n1024

checksum: 8096.8063394986666

```text
         20609 function calls in 0.162 seconds

   Ordered by: cumulative time
   List reduced from 66 to 30 due to restriction <30>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.162    0.162 benchmark_baseline.py:227(checked_operation)
        1    0.000    0.000    0.162    0.162 benchmark_baseline.py:190(compute)
        1    0.001    0.001    0.162    0.162 sample_entropy_profile.py:94(sample_entropy_profile)
        1    0.003    0.003    0.161    0.161 sample_entropy_profile.py:79(_chm)
        2    0.008    0.004    0.111    0.056 sample_entropy_profile.py:68(_mean_cdf)
     2047    0.074    0.000    0.074    0.000 {method 'sort' of 'numpy.ndarray' objects}
     2046    0.000    0.000    0.067    0.000 fromnumeric.py:51(_wrapfunc)
     2044    0.000    0.000    0.066    0.000 fromnumeric.py:1446(searchsorted)
     2044    0.065    0.000    0.065    0.000 {method 'searchsorted' of 'numpy.ndarray' objects}
        3    0.000    0.000    0.043    0.014 _arraysetops_impl.py:145(unique)
        3    0.002    0.001    0.043    0.014 _arraysetops_impl.py:348(_unique1d)
     2044    0.001    0.000    0.037    0.000 fromnumeric.py:968(sort)
        2    0.000    0.000    0.004    0.002 sample_entropy_profile.py:60(_distance_matrix)
        2    0.000    0.000    0.002    0.001 distance.py:2793(cdist)
        2    0.002    0.001    0.002    0.001 {built-in method scipy.spatial._distance_pybind.cdist_chebyshev}
        3    0.001    0.000    0.001    0.000 {method 'flatten' of 'numpy.ndarray' objects}
     2044    0.001    0.000    0.001    0.000 {method 'copy' of 'numpy.ndarray' objects}
        2    0.000    0.000    0.001    0.001 fromnumeric.py:3618(round)
        2    0.001    0.001    0.001    0.001 {method 'round' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 _arraysetops_impl.py:1185(union1d)
     2049    0.000    0.000    0.000    0.000 {built-in method builtins.getattr}
     2055    0.000    0.000    0.000    0.000 {built-in method numpy.asanyarray}
     2044    0.000    0.000    0.000    0.000 fromnumeric.py:1442(_searchsorted_dispatcher)
     2044    0.000    0.000    0.000    0.000 fromnumeric.py:964(_sort_dispatcher)
        1    0.000    0.000    0.000    0.000 validation.py:13(validate_signal)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.empty}
        2    0.000    0.000    0.000    0.000 sample_entropy_profile.py:53(_templates)
        2    0.000    0.000    0.000    0.000 fromnumeric.py:2338(sum)
        2    0.000    0.000    0.000    0.000 _index_tricks_impl.py:813(fill_diagonal)
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
```

## mixed_extractor_n1024

checksum: 1274.8526019227436

```text
         6657 function calls (6599 primitive calls) in 0.019 seconds

   Ordered by: cumulative time
   List reduced from 596 to 30 due to restriction <30>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.019    0.019 benchmark_baseline.py:227(checked_operation)
        1    0.000    0.000    0.019    0.019 benchmark_baseline.py:201(compute)
        1    0.000    0.000    0.019    0.019 extractor.py:423(extract)
        1    0.000    0.000    0.007    0.007 fuzzy.py:60(fuzzy_entropy)
        2    0.005    0.002    0.007    0.003 fuzzy.py:43(_similarity_mean)
        1    0.000    0.000    0.003    0.003 distribution.py:30(distribution_entropy)
        1    0.000    0.000    0.003    0.003 approximate.py:52(approximate_entropy)
        2    0.001    0.000    0.003    0.002 approximate.py:44(_phi)
        1    0.002    0.002    0.003    0.003 _histograms_impl.py:685(histogram)
        2    0.000    0.000    0.002    0.001 distance.py:2793(cdist)
        2    0.002    0.001    0.002    0.001 {built-in method scipy.spatial._distance_pybind.cdist_chebyshev}
        3    0.000    0.000    0.002    0.001 distance.py:1993(pdist)
        3    0.000    0.000    0.002    0.001 _lazy.py:57(lazy_apply)
        3    0.000    0.000    0.002    0.001 _lazy.py:333(wrapper)
        3    0.000    0.000    0.002    0.001 distance.py:2310(_np_pdist)
        3    0.002    0.001    0.002    0.001 {built-in method scipy.spatial._distance_pybind.pdist_chebyshev}
        2    0.000    0.000    0.001    0.001 _axis_nan_policy.py:419(axis_nan_policy_wrapper)
        1    0.000    0.000    0.001    0.001 extractor.py:406(_compute_psd)
        1    0.000    0.000    0.001    0.001 psd.py:236(compute_psd)
        1    0.000    0.000    0.001    0.001 extractor.py:310(_package_version)
        1    0.000    0.000    0.001    0.001 __init__.py:980(version)
        1    0.000    0.000    0.001    0.001 _spectral_py.py:497(welch)
        1    0.000    0.000    0.001    0.001 _spectral_py.py:672(csd)
      153    0.001    0.000    0.001    0.000 {method 'reduce' of 'numpy.ufunc' objects}
        1    0.000    0.000    0.001    0.001 time_domain.py:77(kurtosis)
        1    0.000    0.000    0.001    0.001 _short_time_fft.py:1250(spectrogram)
        1    0.000    0.000    0.001    0.001 __init__.py:483(version)
        1    0.000    0.000    0.001    0.001 _short_time_fft.py:1162(stft_detrend)
        1    0.000    0.000    0.000    0.000 __init__.py:448(metadata)
        1    0.000    0.000    0.000    0.000 __init__.py:954(distribution)
```

## batch_16x256

checksum: 2009.7313151506539

```text
         35860 function calls (35844 primitive calls) in 0.010 seconds

   Ordered by: cumulative time
   List reduced from 242 to 30 due to restriction <30>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.010    0.010 benchmark_baseline.py:227(checked_operation)
        1    0.000    0.000    0.010    0.010 benchmark_baseline.py:216(compute)
        1    0.000    0.000    0.010    0.010 extractor.py:537(extract_batch)
       16    0.000    0.000    0.010    0.001 extractor.py:423(extract)
       16    0.000    0.000    0.007    0.000 extractor.py:310(_package_version)
       16    0.000    0.000    0.007    0.000 __init__.py:980(version)
       16    0.000    0.000    0.005    0.000 __init__.py:483(version)
       16    0.000    0.000    0.005    0.000 __init__.py:448(metadata)
       16    0.000    0.000    0.004    0.000 __init__.py:31(message_from_string)
       16    0.000    0.000    0.004    0.000 parser.py:56(parsestr)
       16    0.000    0.000    0.004    0.000 parser.py:41(parse)
       16    0.000    0.000    0.003    0.000 feedparser.py:173(feed)
       32    0.000    0.000    0.003    0.000 feedparser.py:178(_call_parse)
       32    0.001    0.000    0.003    0.000 feedparser.py:218(_parsegen)
       16    0.000    0.000    0.002    0.000 __init__.py:954(distribution)
       16    0.000    0.000    0.002    0.000 __init__.py:393(from_name)
       16    0.000    0.000    0.001    0.000 {built-in method builtins.next}
     2960    0.000    0.000    0.001    0.000 feedparser.py:129(__next__)
      144    0.000    0.000    0.001    0.000 validation.py:13(validate_signal)
       16    0.000    0.000    0.001    0.000 feedparser.py:472(_parse_headers)
       80    0.000    0.000    0.001    0.000 __init__.py:890(<genexpr>)
       80    0.000    0.000    0.001    0.000 __init__.py:756(search)
       16    0.000    0.000    0.001    0.000 __init__.py:907(read_text)
     2960    0.001    0.000    0.001    0.000 feedparser.py:79(readline)
       16    0.000    0.000    0.001    0.000 _local.py:539(read_text)
       16    0.000    0.000    0.001    0.000 _abc.py:628(read_text)
       32    0.000    0.000    0.001    0.000 {built-in method _io.open}
       16    0.000    0.000    0.001    0.000 _local.py:529(open)
      528    0.000    0.000    0.000    0.000 _policybase.py:301(header_source_parse)
       96    0.000    0.000    0.000    0.000 message.py:500(get)
```
