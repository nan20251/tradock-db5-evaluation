# S-T151.1 top1 example

Positive definition: `fnat > 0.3`.

- Original top1: MODEL 279 / T151_S05.M01, fnat=0.0, DockQ=0.0005, incorrect.
- TraDock top1: MODEL 342 / T151_S12.M04, fnat=0.925, DockQ=0.9026, high.

Why TraDock top1 is success: `fnat=0.925` means the model recovers 92.5% of native interface contacts, which is above the 0.3 threshold. It also has low RMSD (`iRMS=0.6965`, `LRMS=1.7241`), so DockQ is high.
