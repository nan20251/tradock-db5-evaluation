# Portable DB5 Data Parts

This directory stores the DB5 evaluation portable data archive directly in the
git repository as split files. Each part is below 50 MB to avoid GitHub's large
file warning threshold.

Rebuild and restore the data on a new server:

```bash
cd ~/tradock-db5-evaluation
bash scripts/restore_repo_data.sh
```

The script reconstructs:

```text
tradock_db5_eval_pack_20260714_154813_portable.tar.gz
```

Then it verifies:

```text
tradock_db5_eval_pack_20260714_154813_portable.tar.gz.sha256
```

and restores the embedded data archive to:

```text
~/tradock_data/PPCBench
~/tradock_data/autodl-tmp/tools/hdocklite_full
~/tradock_data/TraDock/Trained_models/pretrain_with_sasa/TransformerDock_best.chk
```

The upstream PPCBench source is:

```text
https://github.com/Yukki1777/PPCBench
https://zenodo.org/records/16932314
```
