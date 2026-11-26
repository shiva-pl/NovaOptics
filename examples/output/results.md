| design | mean RMS WFE [waves] | max RMS WFE [waves] | mean RMS spot [um] | max distortion | time [s] |
|---|---|---|---|---|---|
| phase-first, aspheric | 0.102 | 0.157 | 2.9 | 0.51% | 84 |
| phase-first, spherical (asphere fade) | 0.281 | 0.372 | 5.8 | 0.71% | 22 |
| conventional, aspheric | 0.184 | 0.213 | 4.5 | 0.24% | 31 |
| conventional, spherical | 0.311 | 0.431 | 5.9 | 0.80% | 9 |

All figures evaluated on 21 field points (0-20 deg) x 8x16 pupil samples, F/d/C, after optimization on 5 field points x 6x12 pupil samples.

## phase-first, aspheric

architecture ranking in phase space: [('-++', 0.15509337534145307), ('+-+', 0.35777020569012163), ('+++', 0.3699279679205777), ('++-', 0.6555283692802641)]; chosen -++

```
item          R1 [mm]      R2 [mm]   t [mm]  glass                       gap->   max |W| [waves]
E0           -136.977     457.159*    3.217  N-SF66                     10.146              0.00
STOP                                                                     2.682
E1             40.585     -28.572*    8.000  N-PK51                     34.134              0.00
E2            -17.227     -45.673*    2.000  SF56A                       9.204              0.00
* rear surface is an even asphere
```

## phase-first, spherical (asphere fade)

asphere fade of each materialized architecture; best was +-+

```
item          R1 [mm]      R2 [mm]   t [mm]  glass                       gap->   max |W| [waves]
E0             19.544       35.861    7.769  N-LASF41                    1.677              0.00
STOP                                                                     0.398
E1            -34.317       21.799    2.000  SF56A                       2.029              0.00
E2             48.594      -26.926    2.669  N-LASF31A                  40.634              0.00
```

## conventional, aspheric

3/12 Cooke-type starts converged usefully

```
item          R1 [mm]      R2 [mm]   t [mm]  glass                       gap->   max |W| [waves]
STOP                                                                     1.130
E0             31.535     -75.648*    4.195  N-LASF41                    2.525              0.00
E1            -16.532      19.257*    2.000  N-SF5                       1.251              0.00
E2             41.555     -24.267*    5.416  N-LASF44                   45.238              0.00
* rear surface is an even asphere
```

## conventional, spherical

2/12 Cooke-type starts converged usefully

```
item          R1 [mm]      R2 [mm]   t [mm]  glass                       gap->   max |W| [waves]
E0             19.977       42.091    8.000  N-LASF41                    1.970              0.00
STOP                                                                     0.326
E1            -40.258       21.448    2.000  N-SF6                       2.597              0.00
E2             52.109      -29.137    2.645  N-LASF31A                  39.809              0.00
```
