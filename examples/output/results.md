| design | mean RMS WFE [waves] | max RMS WFE [waves] | mean RMS spot [um] | max distortion | time [s] |
|---|---|---|---|---|---|
| phase-first, aspheric | 0.128 | 0.183 | 3.2 | 1.44% | 411 |
| phase-first, spherical (asphere fade) | 0.319 | 0.436 | 6.0 | 0.77% | 138 |
| conventional, aspheric | 0.157 | 0.235 | 4.1 | 0.77% | 85 |
| conventional, spherical | 0.332 | 0.476 | 5.9 | 1.07% | 42 |

All figures evaluated on 21 field points (0-20 deg) x 8x16 pupil samples, F/d/C, after optimization on 5 field points x 6x12 pupil samples.

## phase-first, aspheric

architecture ranking in phase space: [('-++', 0.12828666446702192), ('+-+', 0.210715515160337), ('++-', 0.6617686475877221), ('+++', 2.4379403561902726)]; chosen -++

```
item        R1 [mm]    R2 [mm]  t [mm]                 glass    gap->   max |W| [waves]
E0          -72.674  -800.520*   6.946                 N-SF6    9.877              0.00
STOP                                                           -0.000
E1           43.719   -25.407*   7.475               N-FK51A   40.253              0.00
E2          -18.512   -48.741*   2.000                N-SF11    6.591              0.00
* rear surface is an even asphere
```

## phase-first, spherical (asphere fade)

asphere fade of each materialized architecture; best was +-+

```
item        R1 [mm]    R2 [mm]  t [mm]                 glass    gap->   max |W| [waves]
E0           19.568     41.367   8.000              N-LASF44    2.178              0.00
STOP                                                            0.326
E1          -40.085     20.886   2.000                N-SF11    2.506              0.00
E2           48.636    -30.111   2.625             N-LASF31A   39.369              0.00
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
