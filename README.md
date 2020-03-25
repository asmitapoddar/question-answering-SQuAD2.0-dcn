**Make sure your pushes to master doesn't break any tests.**

# Dynamic Coattention Networks For Question Answering

This repository contain a reimplementation of this paper https://arxiv.org/abs/1611.01604.

The repo name speaks for itself.

No snitching either.

### Existing impl
* Model: https://github.com/atulkum/co-attention/blob/master/code/model.py
* Batcher: https://github.com/atulkum/co-attention/blob/master/code/data_util/data_batcher.py

### Links to our content
* GDrive: https://drive.google.com/open?id=17K0ZFb_OCdvHgSlkNFErzyx--eYZnoiG
* AML GDoc: https://docs.google.com/document/d/1fit7dYVHn0I0PsAA_HCj3AqnxJ7Wzz-78sb--wxKdA4/edit?usp=sharing

### TODOs
- [ ] ![#f03c15](https://placehold.it/15/f03c15/000000?text=+) *Add your **past** contributions or **nearest-future work** here. (everyone)*
- [x] Move tests to seperate file (Richie)
- [x] Implement model (everyone)
- [x] Enable cuda usage (Kuba)
- [x] Get forward pass to run (Kuba)
- [x] Get backward pass to run (Kuba -- this was quick)
- [x] Debug why predicted end indices are all 0 (Richie)
- [x] Complete batching (Asmita)
- [ ] Training pipeline (Asmita + Kuba's minor cleanup)
- [x] Model serialisation (~~Kuba~~ / Richie :p)
- [ ] Run Training on real data ( ? )
- [ ] Generate predictions for evaluation (TODO batching if needed, better conversion from tokens to answer strings, ~~load serialised model~~) (Dip)
