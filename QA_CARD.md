# Q&A card — the questions that decide the room

Keep this open during judging. Each answer is short on purpose; the number is the argument.

---

**"Your macro-F1 is 0.56 to 0.62. Published systems report 0.90."**

Those headlines are support-weighted, so N2 dominates them. The Dreem Open Dataset paper's own
per-stage F1 values average 82.4 against its 89.9 headline. Our comparable figure is 0.852.
Person-averaged macro-F1 weights N1 the same as N2, and six of our twenty test nights contain no
reference N3 at all, which caps the person-averaged N3 at 0.70 for any model. Compare like with
like and the gap closes.

---

**"Is 0.661 an independent result?"**

No. The BOAS cohort was inspected at v1, so that is a development comparison. The independent
test is Sleep-EDF, and we fail on it. We say so on the slide, in the handout and in the README.

---

**"Why is your best model the one you are warning us about?"**

Because that is the finding. The CNN reached 0.728 on the headband and 0.086 on Sleep-EDF —
the best in-domain model was the worst out-of-domain one. If you pick by in-domain score you
pick the most fragile system. That is the argument for a second cohort rather than a single
accuracy number.

---

**"How do we know you did not tune on the test set?"**

The split is a file, `protocol.json`: 48 fitting people, six selection, six calibration, twenty
held out. Selection code reads the six. The evaluation file cannot fit a model, a calibrator or
a threshold — there is no code in it that could. One mask, `(y >= 0) & (y < 5) & good`, is used
by every scorer. And `reproduce_v2.py` re-runs the whole thing offline in thirty seconds against
a recorded model hash.

---

**"What did you get wrong?"**

Three claims were withdrawn because they fell inside the 0.015 macro-F1 spread we measured
across random seeds. A leakage audit merged two duplicated studies and removed a win we had
claimed in the research track. Prediction intervals missed their 90% target, so the range claim
is switched off rather than softened.

---

**"Did you try a bigger or end-to-end model?"**

Yes, twice, with two samplers that share no code path. End-to-end scored macro-F1 0.434 and
0.435 against 0.618 for the frozen trunk with a temporal head — a gap of 0.184, about thirteen
times the seed spread. With 48 people, freezing the trunk is a constraint and the constraint is
what helps. The limit is the number of people, not the number of epochs.

---

**"Does it work in real time?"**

Yes, and we measured the cost: 0.038 kappa against the same model scored retrospectively on the
same people. A trailing window replaces whole-recording normalisation and the smoother runs
forward only. `live_inference.py --replay` shows it epoch by epoch.

---

**"What would you need next?"**

More people, not more epochs or more nights from the same device — the end-to-end result says
that directly. A second device cohort with its own labels would let us test whether the
transfer failure is specific to Sleep-EDF or general.

---

**"Can we use this clinically?"**

No, and nothing in the package claims otherwise. No clinical, ADHD, medication or paediatric
claim is made. The delivery-research track records what would have to be measured first —
entry biomarkers, outcome scales and safety monitoring per route — explicitly as an unmet
evidence burden, not as work we did.
