| design | mean RMS WFE [waves] | max RMS WFE [waves] | time [s] |
|---|---|---|---|
| 3 elements (start) | 0.281 | 0.372 | 0 |
| +1 air-spaced element | 0.178 | 0.235 | 707 |
| cemented (behind surface 1) | 0.289 | 0.389 | 510 |

f = 50 mm, F/4, +/-20 deg, F/d/C; dense 21-field grid; preferred SCHOTT glasses.

## 3 elements (start)

```
surf      R [mm]   t [mm]  material    label
0         19.544    7.769  N-LASF41    E0 front
1         35.861    1.677  air         E0 rear
2            inf    0.398  air         stop  (stop)
3        -34.317    2.000  SF56A       E1 front
4         21.799    2.029  air         E1 rear
5         48.594    2.669  N-LASF31A   E2 front
6        -26.926   40.634  air         E2 rear
7            inf    0.000  air         image
```

## +1 air-spaced element

```
surf      R [mm]   t [mm]  material    label
0         21.162    3.038  N-LASF41    E0 front
1         35.805    6.610  air         E0 rear
2            inf    0.523  air         stop  (stop)
3        -29.466    2.000  SF56A       E1 front
4         33.161    1.560  air         E1 rear
5        165.574    4.997  N-LASF31A   E2 front
6        -25.016    1.000  air         E2 rear
7         26.665    8.000  N-LAK33B    new front
8         29.515   35.425  air         new rear
9            inf    0.000  air         image
```

## cemented (behind surface 1)

```
surf      R [mm]   t [mm]  material    label
0         19.605    6.002  N-LASF41    E0 front
1         34.717    2.000  N-LAK14     cemented
2         37.239    1.600  air         E0 rear
3            inf    0.387  air         stop  (stop)
4        -35.042    2.000  SF56A       E1 front
5         21.695    2.132  air         E1 rear
6         49.074    2.595  N-LASF31A   E2 front
7        -27.312   40.431  air         E2 rear
8            inf    0.000  air         image
```
