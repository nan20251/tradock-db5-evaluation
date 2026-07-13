# S-T041.1 TraDock top1 failure example

Positive definition: `fnat > 0.3`.

This is a useful failure case because the original top1 is already a successful pose, but TraDock reranking moves an incorrect pose to rank 1.

| structure | MODEL | identification | TraDock score | fnat | DockQ | iRMS | LRMS | result |
|---|---:|---|---:|---:|---:|---:|---:|---|
| original top1 | 328 | T41_S01.M01 | -0.7157 | 0.8305 | 0.5976 | 1.7988 | 5.5165 | success |
| best available by fnat | 376 | T41_S09.M01 | -0.7236 | 0.9492 | 0.9255 | 0.6413 | 1.1361 | success/high |
| TraDock top1 | 408 | T41_S14.M03 | -0.6990 | 0.0000 | 0.0018 | 38.3031 | 75.8587 | failure |

Why the TraDock top1 fails: its `fnat=0.0`, so it recovers none of the native interface contacts and is far below the `fnat > 0.3` success threshold. Its RMSD values are also very large, so DockQ is nearly zero.

Failure type: score misranking. A good pose exists and even the original top1 is good, but the MDN score ranks an incorrect pose higher.
