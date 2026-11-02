---
paths:
  - "tools/**"
  - "figures/**"
---
# Physical units in figures

- Z (geopotential) is plotted as geopotential in m²/s², exactly as stored. Never divide by g to show
  geopotential height (gpm/dam) and never relabel it as height: the user corrected this explicitly
  (2026-10-15, `tools/plot_ens_anom_gaussianity.py` PHYS table).
- In general, show variables in the quantity they are; only rescale within the same quantity
  (Pa -> hPa, kg/kg -> g/kg).
