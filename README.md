# PPDPP
## Re-DocRed fusion utility

This repo now includes `docred_debate_fusion.py`, a lightweight debate-style
fusion script for combining `predict-LLM-converted.json` and
`results-SLM.json` on `test_revised.json`. It implements a symbolic policy
selector inspired by the original PPDPP environment/action/reward setup and
writes both fused predictions and an evaluation report.

Example:

```bash
python docred_debate_fusion.py \
  --dataset test_revised.json \
  --llm predict-LLM-converted.json \
  --slm results-SLM.json \
  --output fused-debate.json \
  --report fused-debate-report.json
```
