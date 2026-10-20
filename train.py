"""SpamSentry — SMS/text spam detector.
Trains TF-IDF + MultinomialNB on a built-in labeled SMS dataset and
exports the model weights (vocab, idf, class priors, feature log-probs)
as JSON so the browser demo reproduces sklearn's predict_proba exactly,
with no backend.
"""
import json
import math
import random

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

rng = random.Random(42)

# ---- Synthetic SMS dataset (spam vs ham, rng seed 42) ----
PRIZE = ["iPhone 15", "holiday", "gift card", "laptop", "flat-screen TV",
         "smartwatch", "cash bonus"]
AMOUNT = ["$1000", "$5000", "$250", "$10000"]
CODE = ["80808", "12345", "99999", "55555"]
PHONE = ["0900-123-456", "1-800-WINNER", "0900-555-0199"]
SITE = ["prizes-claim.com", "win-now.net", "freegifts.io"]
LOTTO = ["Mega", "National", "Euro"]
COMPANY = ["TelcoPlus", "MobileRewards", "PrizeLine"]

SPAM_TPL = [
    "WINNER!! You have been selected to receive a {prize}. Call {phone} now to claim your prize",
    "Congratulations! You have won a free {prize} worth {amount}. Reply WIN to {code} to claim now",
    "URGENT: Your mobile number won {amount} in the {lotto} lottery draw. Text CLAIM to {code}",
    "Free entry to win {amount} cash! Text YES to {code}. Terms apply. Reply STOP to opt out",
    "You have 1 new voicemail from an unknown prize line. Call {phone} now, free minutes for winners",
    "Limited offer: get a free {prize} today only. Visit {site} and claim before midnight",
    "Dear customer, your account won {amount}. Confirm your details at {site} to receive payment",
    "CONGRATS! Claim your free {prize} now. Text CLAIM to {code}. Message rates may apply",
    "You are our lucky winner! {amount} cash prize waiting. Call {phone} to collect today",
    "Exclusive deal: {prize} at 90 percent off, today only. Order now at {site} before stock runs out",
    "Final notice: your {amount} reward expires today. Text REWARD to {code} to claim it now",
    "Hi, this is {company} promotions. You won a free {prize}. Reply YES to {code} to confirm",
]

PLAN = ["dinner", "coffee", "lunch", "the movie"]
TIME = ["7pm", "6:30", "noon", "8pm"]
ITEM = ["milk", "bread", "groceries", "the keys", "your charger"]
TRANSPORT = ["bus", "train", "metro"]
MINS = ["10", "15", "20"]
DAY = ["Sunday", "Saturday", "Friday"]
EVENT = ["meeting", "birthday party", "study session"]
TASK = ["work", "the gym", "class"]
TEAM = ["football", "cricket"]
DOC = ["notes", "report", "photos"]
PURPOSE = ["class", "work", "the project"]
PLACE = ["station", "mall", "cafe"]

HAM_TPL = [
    "Hey, are we still meeting for {plan} at {time}?",
    "Can you pick up {item} on your way home please?",
    "Running late, the {transport} is delayed. See you in {mins} minutes",
    "Happy birthday! Hope you have a wonderful day",
    "Mom called, she wants us to visit on {day}",
    "Don't forget the {event} tomorrow at {time}. Bring your {item}",
    "Just finished {task}, heading home now. Want anything from the store?",
    "The {team} match is on tonight at {time}, come over if you want to watch",
    "Can you send me the {doc} you mentioned yesterday? Need it for {purpose}",
    "Lunch was great, thanks! Let's do it again next {day}",
    "I'll be at the {place} around {time}, call me when you arrive",
    "Reminder: dentist appointment on {day} at {time}, don't be late",
]


def fill(tpl, spam):
    if spam:
        return tpl.format(
            prize=rng.choice(PRIZE), amount=rng.choice(AMOUNT), code=rng.choice(CODE),
            phone=rng.choice(PHONE), site=rng.choice(SITE), lotto=rng.choice(LOTTO),
            company=rng.choice(COMPANY))
    return tpl.format(
        plan=rng.choice(PLAN), time=rng.choice(TIME), item=rng.choice(ITEM),
        transport=rng.choice(TRANSPORT), mins=rng.choice(MINS), day=rng.choice(DAY),
        event=rng.choice(EVENT), task=rng.choice(TASK), team=rng.choice(TEAM),
        doc=rng.choice(DOC), purpose=rng.choice(PURPOSE), place=rng.choice(PLACE))


SPAM = [fill(t, True) for t in SPAM_TPL for _ in range(9)]
HAM = [fill(t, False) for t in HAM_TPL for _ in range(9)]
rng.shuffle(SPAM)
rng.shuffle(HAM)

texts = SPAM + HAM
labels = [1] * len(SPAM) + [0] * len(HAM)   # 1 = spam, 0 = ham
print(f"dataset: {len(texts)} messages ({len(SPAM)} spam, {len(HAM)} ham)")


def make_vectorizer():
    return TfidfVectorizer(
        stop_words="english", ngram_range=(1, 1),
        max_features=200, sublinear_tf=True)


X = np.array(texts)
y = np.array(labels)

# Stratified 5-fold CV (honest metric), then final model on ALL data.
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_accs = []
for tri, tei in skf.split(X, y):
    v = make_vectorizer()
    Xtr = v.fit_transform(X[tri])
    Xte = v.transform(X[tei])
    c = MultinomialNB()
    c.fit(Xtr, y[tri])
    fold_accs.append(accuracy_score(y[tei], c.predict(Xte)))
acc = float(np.mean(fold_accs))
print(f"5-fold CV accuracy: {acc:.3f} (folds={['%.3f' % a for a in fold_accs]})")

vectorizer = make_vectorizer()
Xtr = vectorizer.fit_transform(texts)
clf = MultinomialNB()
clf.fit(Xtr, y)
print(f"trained on {Xtr.shape[0]} samples, {Xtr.shape[1]} features")

feature_names = vectorizer.get_feature_names_out()

weights = {
    "vocab": feature_names.tolist(),
    "idf": [round(float(v), 6) for v in vectorizer.idf_],
    # rows: [ham (class 0), spam (class 1)]
    "class_log_prior": [round(float(v), 6) for v in clf.class_log_prior_],
    "feature_log_prob": [
        [round(float(v), 6) for v in row] for row in clf.feature_log_prob_
    ],
    "accuracy": round(float(acc), 4),
    "fold_accuracies": [round(float(a), 4) for a in fold_accs],
    "n_spam": len(SPAM),
    "n_ham": len(HAM),
    "stop_words": sorted(ENGLISH_STOP_WORDS),
}

with open("weights.json", "w") as f:
    json.dump(weights, f)
print(f"exported weights.json ({len(feature_names)} features)")

# Sanity check: re-implement the JS-side math in pure Python and verify it
# matches sklearn's predict_proba bit-for-bit on sample messages.
STOP = set(ENGLISH_STOP_WORDS)
import re
IDX = {w: i for i, w in enumerate(feature_names)}


def js_predict_proba(text):
    toks = [t for t in re.findall(r"(?u)\b\w\w+\b", text.lower()) if t not in STOP]
    tf = {}
    for t in toks:
        tf[t] = tf.get(t, 0) + 1
    feats = []
    for tok, cnt in tf.items():
        i = IDX.get(tok)
        if i is None:
            continue
        feats.append((i, (1.0 + math.log(cnt)) * weights["idf"][i]))
    norm = math.sqrt(sum(x * x for _, x in feats)) or 1.0
    jll = list(weights["class_log_prior"])
    for i, x in feats:
        v = x / norm
        jll[0] += v * weights["feature_log_prob"][0][i]
        jll[1] += v * weights["feature_log_prob"][1][i]
    m = max(jll)
    e = [math.exp(z - m) for z in jll]
    s = sum(e)
    return [e[0] / s, e[1] / s]


samples = [
    "WINNER!! You won a free iPhone, text CLAIM to 80808 now",
    "Hey, are we still meeting for dinner at 7pm?",
    "Free entry to win $5000 cash, text YES now",
    "Can you pick up milk on your way home please?",
]
ok = True
for s in samples:
    mine = js_predict_proba(s)
    sk = clf.predict_proba(vectorizer.transform([s]))[0]
    diff = max(abs(mine[0] - sk[0]), abs(mine[1] - sk[1]))
    ok = ok and diff < 1e-5
    print(f"parity check: {s[:45]!r} -> mine={mine[1]:.6f} sklearn={sk[1]:.6f} diff={diff:.2e}")
print("JS parity:", "PASS" if ok else "FAIL")

# Most spammy / hammy words as a sanity check
order = np.argsort(clf.feature_log_prob_[1] - clf.feature_log_prob_[0])
print("most hammy:", [feature_names[i] for i in order[:8]])
print("most spammy:", [feature_names[i] for i in order[-8:][::-1]])
