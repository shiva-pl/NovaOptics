| design | mean RMS WFE [waves] | max RMS WFE [waves] | time [s] |
|---|---|---|---|
| 3 elements (start) | 0.281 | 0.372 | 0 |
| +1 air-spaced element | 0.178 | 0.235 | 397 |
| cemented (behind surface 1) | 0.227 | 0.280 | 1581 |

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
0         21.661    7.469  N-LASF41    E0 front
1      -3118.481    2.000  F2G12       cemented
2         35.417    1.602  air         E0 rear
3            inf    0.328  air         stop  (stop)
4        -38.137    2.000  SF56A       E1 front
5         22.276    2.041  air         E1 rear
6         48.195    4.835  N-LASF31A   E2 front
7        -30.451   38.723  air         E2 rear
8            inf    0.000  air         image
```
