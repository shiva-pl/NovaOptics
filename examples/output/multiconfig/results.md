## zoom 35-70 mm

81 s

| configuration | EFL [mm] | RMS WFE per field [waves] | image distance [mm] |
|---|---|---|---|
| wide | 35.00 | 1.275 1.421 1.606 | 42.914 |
| mid | 50.00 | 0.625 0.621 0.637 | 42.914 |
| tele | 69.99 | 0.429 0.861 1.287 | 42.914 |

```
components:
#  component label         surfaces   power [1/m]  details
0  lens      G1            0-1              +7.11  R 51.86 / 121.04  t 8.00  N-PSK53A  -> 10.000 mm
1  lens      G2            2-3             -16.99  R -68.70 / 84.80  t 2.63  N-LAK21  -> 10.000 mm
2  stop                    4-4                       -> 0.959 mm
3  lens      G3            5-6             +27.34  R 35.04 / -58.15  t 8.00  N-PSK53A  -> 42.914 mm
4  image     image         7-7                     

surfaces:
surf type          R [mm]    t [mm]  material          conic  label
0    conic        51.8581    8.0000  N-PSK53A              0  G1
1    conic       121.0351   10.0000  air                   0
2    conic       -68.6965    2.6295  N-LAK21               0  G2
3    conic        84.7980   10.0000  air                   0
4    conic            inf    0.9587  -                     0   STOP
5    conic        35.0383    8.0001  N-PSK53A              0  G3
6    conic       -58.1534   42.9142  air                   0
7    conic            inf    0.0000  -                     0  image

configurations:
  1 wide: 1:t = 5.9413, 3:t = 51.5562 EFL 35
  2 mid: 1:t = 26.9664, 3:t = 30.5312 EFL 50
  3 tele: 1:t = 39.7550, 3:t = 0.0410 EFL 70
```

## spectrum from defocus

39 s

| configuration | EFL [mm] | RMS WFE per field [waves] | image distance [mm] |
|---|---|---|---|
| 450 nm | 48.98 | 0.064 0.119 | 47.200 |
| 550 nm | 49.62 | 0.171 0.143 | 47.600 |
| 650 nm | 50.00 | 0.125 0.101 | 48.000 |
| 750 nm | 50.26 | 0.008 0.035 | 48.400 |
| 850 nm | 50.45 | 0.120 0.146 | 48.800 |

```
components:
#  component label         surfaces   power [1/m]  details
0  stop                    0-0                       -> 46.314 mm
1  lens      L1            1-3             +20.09  R 38.50 / -14.47 / -103.11  t 2.00 + 2.01  N-SK16 | N-LAK7  cemented  -> 48.000 mm
2  image     image         4-4                     

surfaces:
surf type          R [mm]    t [mm]  material          conic  label
0    conic            inf   46.3143  -                     0   STOP
1    conic        38.5005    2.0000  N-SK16                0  L1
2    conic       -14.4681    2.0066  N-LAK7                0  L2
3    conic      -103.1061   48.0000  air                   0
4    conic            inf    0.0000  -                     0  image

configurations:
  1 450 nm: 3:t = 47.2000 wavelengths 450 nm (no EFL target)
  2 550 nm: 3:t = 47.6000 wavelengths 550 nm (no EFL target)
  3 650 nm: 3:t = 48.0000 wavelengths 650 nm
  4 750 nm: 3:t = 48.4000 wavelengths 750 nm (no EFL target)
  5 850 nm: 3:t = 48.8000 wavelengths 850 nm (no EFL target)
```

