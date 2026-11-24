| design | mean RMS WFE [waves] | max RMS WFE [waves] | mean RMS spot [um] | max distortion | time [s] |
|---|---|---|---|---|---|
| phase-first, aspheric | 0.122 | 0.182 | 3.5 | 0.83% | 449 |
| phase-first, spherical (asphere fade) | 0.281 | 0.372 | 5.8 | 0.71% | 158 |
| conventional, aspheric | 0.174 | 0.233 | 4.4 | 0.65% | 91 |
| conventional, spherical | 0.311 | 0.431 | 5.9 | 0.80% | 60 |

All figures evaluated on 21 field points (0-20 deg) x 8x16 pupil samples, F/d/C, after optimization on 5 field points x 6x12 pupil samples.

## phase-first, aspheric

architecture ranking in phase space: [('-++', 0.15509337534138173), ('+-+', 0.3577702056890844), ('+++', 0.3699279679140329), ('++-', 0.6555283692820657)]; chosen -++

```
item          R1 [mm]      R2 [mm]   t [mm]  glass                       gap->   max |W| [waves]
E0            -59.818    -435.642*    2.000  N-SF66                      8.094              0.00
STOP                                                                    -0.000
E1             46.954     -25.728*    8.000  N-SK5                      34.004              0.00
E2            -16.728     -39.162*    2.000  SF6G05                     10.179              0.00
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
E0             24.447    -101.017*    4.153  N-LASF44                    0.657              0.00
STOP                                                                     1.418
E1            -19.574      13.758*    2.000  N-SF2                       1.292              0.00
E2             21.996     -45.044*    7.083  N-LASF44                   41.921              0.00
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
