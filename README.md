# SpamSentry 🛡️

ML-powered **SMS spam detector** — paste any text message and a real trained model tells you if it's SPAM or HAM, right in the browser.

**Live demo:** https://spamsentry-lyart.vercel.app

## How it works

1. **Data** — 216 labeled SMS messages (108 spam / 108 ham) generated from class-controlled templates in `train.py` (random seed 42).
2. **Features** — TF-IDF unigrams (sublinear TF, L2-normalized, top 200 features) via scikit-learn.
3. **Model** — Multinomial Naive Bayes. **5-fold cross-validation accuracy: 100%.**
4. **Browser demo** — `train.py` exports the vocabulary, idf values, class priors, feature log-probabilities, and stop-word list to `weights.json`. The site's JavaScript re-runs the exact same math as sklearn's `predict_proba` (joint log-likelihood + softmax) — verified to match Python's probabilities to 8 decimals on test messages. No backend, no API calls.

## Reproduce

```bash
pip install scikit-learn numpy
python train.py   # retrains and regenerates weights.json
```

## Files

- `train.py` — dataset generation, training, evaluation, weight export (+ JS parity check)
- `weights.json` — exported model (169 features)
- `index.html` — the interactive demo site
- `make_ppt.py` — builds the project presentation (python-pptx)
- `vercel.json` — static hosting config + security headers

Built by [Vikranth Reddy](https://github.com/HVRGAMING2705) · B.Tech CSE (AI & ML), Woxsen University
