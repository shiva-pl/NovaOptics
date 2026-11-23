| design | mean RMS WFE [waves] | max RMS WFE [waves] | mean RMS spot [um] | max distortion | time [s] |
|---|---|---|---|---|---|
| phase-first, aspheric | 0.186 | 0.260 | 4.3 | 0.71% | 243 |
| phase-first, spherical (asphere fade) | 0.344 | 0.444 | 7.6 | 0.87% | 83 |
| conventional, aspheric | 0.157 | 0.235 | 4.1 | 0.77% | 41 |
| conventional, spherical | 0.332 | 0.476 | 5.9 | 1.07% | 21 |

All figures evaluated on 21 field points (0-20 deg) x 8x16 pupil samples, F/d/C, after optimization on 5 field points x 6x12 pupil samples.

## phase-first, aspheric

architecture ranking in phase space: [('-++', 0.12828666446706896), ('+-+', 0.21071551516083603), ('++-', 0.6617686475849636), ('+++', 2.437940356190656)]; chosen +-+

```
item        R1 [mm]    R2 [mm]  t [mm]                 glass    gap->   max |W| [waves]
E0           22.851    38.077*   8.000             N-LASF31A    0.973              0.00
STOP                                                            1.967
E1          -14.908    80.385*   2.000                N-SF11    1.384              0.00
E2           46.413   -22.802*   2.977             N-LASF31A   43.170              0.00
* rear surface is an even asphere
```

## phase-first, spherical (asphere fade)

asphere fade of each materialized architecture; best was +-+

```
item        R1 [mm]    R2 [mm]  t [mm]                 glass    gap->   max |W| [waves]
E0           19.942    -89.285   4.105              N-LASF44    0.944              0.00
STOP                                                            0.557
E1          -29.177     20.397   2.000                 N-SF1    2.709              0.00
E2          -41.880    -19.907   3.874             N-LASF31A   41.584              0.00
```

## conventional, aspheric

2/12 Cooke-type starts converged usefully

```
item        R1 [mm]    R2 [mm]  t [mm]                 glass    gap->   max |W| [waves]
E0           27.131   -92.875*   4.874             N-LASF31A   -0.000              0.00
STOP                                                            1.876
E1          -20.621    15.268*   2.639                 N-SF1    1.214              0.00
E2           23.732   -49.828*   7.250             N-LASF31A   41.460              0.00
* rear surface is an even asphere
```

## conventional, spherical

2/12 Cooke-type starts converged usefully

```
item        R1 [mm]    R2 [mm]  t [mm]                 glass    gap->   max |W| [waves]
E0           20.831     51.933   8.000             N-LASF31A    2.321              0.00
STOP                                                            0.237
E1          -52.307     21.024   2.000                N-SF57    3.619              0.00
E2           58.298    -32.329   2.667             N-LASF31A   38.764              0.00
```
