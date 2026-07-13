# TraDock / BioScore Runbook

This file is a compact operational guide for continuing the TraDock and
BioScore-PPI comparison.

## 1. Resume Context In A New Codex Session

Say:

```text
Read /Users/nanyang/Documents/AutoDL-Projects/TraDock/PROJECT_MEMORY.md and RUNBOOK.md, then continue the TraDock/BioScore task.
```

Do not paste passwords into files. Use interactive SSH password prompts when
needed.

## 2. Connect To Servers

TraDock:

```bash
ssh -p 19464 root@connect.bjb1.seetacloud.com
cd /root/TraDock
```

BioScore:

```bash
ssh -p 46984 root@connect.westc.seetacloud.com
cd /root/BioScore-PPI
```

## 3. Check TraDock Training State

```bash
cd /root/TraDock
pid=$(pgrep -f 'python .*examples/train.py' | head -1)
echo train_pid=$pid
if [ -n "$pid" ]; then
  ps -o pid,ppid,stat,etime,pcpu,pmem,rss,cmd -p "$pid"
fi
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw --format=csv,noheader

LOG=$(ls -t /root/autodl-tmp/tradock_*with_sasa_*.log 2>/dev/null | head -1)
echo log=$LOG
if [ -n "$LOG" ]; then
  grep -niE 'nan|inf|error|traceback|cuda out|oom|exception|killed' "$LOG" | tail -40 || true
  tail -80 "$LOG"
fi
```

Check best checkpoint:

```bash
cd /root/TraDock
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python - <<'PY'
import os, time, torch
p = 'Trained_models/pretrain_with_sasa/TransformerDock_best.chk'
ckpt = torch.load(p, map_location='cpu', weights_only=False)
print('mtime', time.strftime('%F %T', time.localtime(os.path.getmtime(p))))
print('epoch', ckpt.get('epoch'))
print('loss', ckpt.get('loss'))
PY
```

## 4. Current TraDock Best Model

Use:

```text
/root/TraDock/Trained_models/pretrain_with_sasa/TransformerDock_best.chk
```

Expected metadata:

```text
epoch = 27
loss = 0.7242806092681128
```

## 5. Check TraDock Evaluation Progress

Current full 113 fnat>0.3 evaluation:

```text
/root/TraDock/results/capri_eval_113_fnat03_best_epoch027_single.csv
/root/TraDock/results/capri_eval_113_fnat03_best_epoch027_single.summary.csv
/root/autodl-tmp/tradock_eval_fnat03_single_20260619_172127.log
```

Status command:

```bash
cd /root/TraDock
pid=$(pgrep -f 'examples/eval_capri_fast.py' | head -1)
echo eval_pid=$pid
if [ -n "$pid" ]; then
  ps -o pid,ppid,stat,etime,pcpu,pmem,rss,cmd -p "$pid"
fi
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw --format=csv,noheader

LOG=$(ls -t /root/autodl-tmp/tradock_eval_fnat03_single_*.log | head -1)
echo log=$LOG
stat -c 'log_size=%s mtime=%y' "$LOG"
grep -niE 'warn|warning|error|traceback|exception|nan|inf|skip|跳过|失败|cuda out|oom|killed' "$LOG" | tail -80 || true
tail -140 "$LOG"
```

CSV progress and partial metrics:

```bash
cd /root/TraDock
python - <<'PY'
import csv, os, statistics
summary = 'results/capri_eval_113_fnat03_best_epoch027_single.summary.csv'
detail = 'results/capri_eval_113_fnat03_best_epoch027_single.csv'

if os.path.exists(summary):
    with open(summary, newline='') as f:
        rows = list(csv.DictReader(f))
    done = [r for r in rows if r.get('status') == 'done']
    bad = [r for r in rows if r.get('status') not in ('done', 'skipped', 'skip')]
    print('summary_rows', len(rows), 'done_targets', len(done),
          'last_done', done[-1]['target'] if done else None,
          'bad_status', len(bad))
    for r in rows[-6:]:
        print(r['target_index'], r['target'], r['status'],
              'models', r.get('n_models'),
              'auc', r.get('auc'),
              'sp', r.get('spearman'),
              's1', r.get('success@1'),
              's5', r.get('success@5'),
              's10', r.get('success@10'))

    def fnum(x):
        try:
            return float(x)
        except Exception:
            return None

    aucs = [fnum(r.get('auc_pos') or r.get('auc')) for r in done]
    aucs = [x for x in aucs if x is not None]
    print('partial_mean_auc', statistics.mean(aucs) if aucs else None)

    for k in [1, 2, 5, 10, 100]:
        vals = [int(float(r.get(f'success@{k}', 0) or 0)) for r in done]
        print(f'partial_success@{k}', sum(vals), '/', len(done),
              '=', (sum(vals) / len(done) if done else None))

if os.path.exists(detail):
    with open(detail, newline='') as f:
        drows = list(csv.DictReader(f))
    print('detail_rows', len(drows))
PY
```

ETA estimate:

```bash
cd /root/TraDock
python - <<'PY'
import csv, os, statistics, time, re
summary = 'results/capri_eval_113_fnat03_best_epoch027_single.summary.csv'
log = os.popen("ls -t /root/autodl-tmp/tradock_eval_fnat03_single_*.log | head -1").read().strip()
print('now', time.strftime('%F %T'))
print('log', log)

if os.path.exists(summary):
    with open(summary, newline='') as f:
        rows = list(csv.DictReader(f))
    done = [r for r in rows if r.get('status') == 'done']
    secs = []
    for r in done:
        try:
            secs.append(float(r.get('elapsed_sec') or 0))
        except Exception:
            pass

    total = 113
    rem = total - len(done)
    if secs:
        avg = statistics.mean(secs)
        med = statistics.median(secs)
        avg10 = statistics.mean(secs[-10:] if len(secs) >= 10 else secs)
        print('done', len(done), '/', total, 'remaining', rem,
              'last_done', done[-1]['target'] if done else None)
        print('avg_sec_per_target', avg, 'median', med, 'avg_last10', avg10)
        for label, rate in [('overall_avg', avg), ('last10_avg', avg10), ('median', med)]:
            eta = rem * rate
            print(label, 'eta_hours', eta / 3600,
                  'eta_finish', time.strftime('%F %T', time.localtime(time.time() + eta)))

if log and os.path.exists(log):
    current = None
    for line in open(log):
        m = re.match(r'\[(\d+)/(\d+)\]\s+(\S+)', line)
        if m:
            current = m.groups()
    print('current_target', current)
PY
```

## 6. Restart TraDock Evaluation If Needed

Use single worker unless `eval_capri_fast.py` multiprocessing is fixed.

```bash
cd /root/TraDock
OUT=results/capri_eval_113_fnat03_best_epoch027_single.csv
LOG=/root/autodl-tmp/tradock_eval_fnat03_single_$(date +%Y%m%d_%H%M%S).log

nohup env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python -u examples/eval_capri_fast.py \
    --data_dir data/database \
    --checkpoint Trained_models/pretrain_with_sasa/TransformerDock_best.chk \
    --out "$OUT" \
    --pos_metric fnat \
    --pos_threshold 0.3 \
    --success_denominator all \
    --score_type mdn \
    --n_workers 1 \
    --resume \
    > "$LOG" 2>&1 &
```

Use `--resume` only when continuing the same output file and summary CSV.

## 7. After TraDock Evaluation Finishes

Check aggregate summary:

```bash
cd /root/TraDock
ls -lh results/capri_eval_113_fnat03_best_epoch027_single*
tail -20 /root/autodl-tmp/tradock_eval_fnat03_single_*.log
cat results/capri_eval_113_fnat03_best_epoch027_single.aggregate.csv 2>/dev/null || true
```

If no aggregate CSV is generated, compute from summary CSV using the same
fields:

- `mean_auc_pos`
- `mean_spearman`
- `mean_pearson`
- `mean_top20_spearman`
- `success@1`, `success@2`, `success@5`, `success@10`, `success@100`
- denominator: all 113 targets

## 8. BioScore Dedup Result Reference

Dedup BioScore full 113 true `fnat > 0.3`, high score better:

```text
Mean AUC fnat>0.3: 0.46347448735628816
Success@1: 18 / 113 = 15.93%
Success@2: 28 / 113 = 24.78%
Success@5: 38 / 113 = 33.63%
Success@10: 44 / 113 = 38.94%
Success@100: 72 / 113 = 63.72%
```

Original non-dedup BioScore full 113 true `fnat > 0.3`, high score better:

```text
Mean AUC fnat>0.3: 0.5414061880452503
Success@1: 27 / 113 = 23.89%
Success@2: 34 / 113 = 30.09%
Success@5: 44 / 113 = 38.94%
Success@10: 51 / 113 = 45.13%
Success@100: 75 / 113 = 66.37%
```

## 9. Compare TraDock And BioScore

After TraDock full 113 finishes, compare:

- TraDock dedup, epoch 27, full 113, true `fnat > 0.3`
- BioScore dedup, full 113, true `fnat > 0.3`
- Optionally BioScore non-dedup as raw/leaky baseline only

Use all 113 as denominator for Success.

## 10. Important Evaluation Warnings

- PyTorch warning about `enable_nested_tensor` is expected and harmless.
- Any `Traceback`, `CUDA out of memory`, `nan`, `inf`, or `failed` target should be investigated.
- If `--n_workers 4` stalls with no CSV rows, kill the parent and orphan workers, then restart with `--n_workers 1`.

