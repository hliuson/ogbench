# Runs

Preliminary sweep: 

```bash
python submit_100m_reference_suite_slurm.py \
  --task puzzle-4x6 \
  --task cube-triple \
  --task humanoidmaze-giant-navigate \
  --method gciql \
  --method trl \
  --method latent \
  --seed 0 \
  --tag prelim \
  --submit
```
