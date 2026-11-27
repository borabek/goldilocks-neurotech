# NeuroRelease research track

The drug-delivery half of the hackathon project. It is separate from the EEG sleep-staging work:
no neural or physiological signal enters here.

- `exposure_model/`: predicting brain exposure from blood exposure, formulation and route, tested
  on source studies held out from training (`EXPOSURE_RESULT.md`).
- `formulation_evidence/`: the formulation and host-factor evidence tables, with the rules that
  decide which questions the evidence can and cannot answer, and their tests.
- `coating_core/`: notes and values on coating and core choice.

Source datasets and their licences are listed in [`docs/DATA_SOURCES.md`](../docs/DATA_SOURCES.md)
and inside each source note. Some sources are licensed for non-commercial use only.
