# QA validation suite

The repository contains established `src.preprocessing` and `src.prediction`
interfaces, but no serialized model artifact or manifest. The tests therefore:

* exercise the current input validator when pandas is available;
* check the documented input and CSV contracts using synthetic records;
* verify the source CSV header without emitting customer rows; and
* provide explicit skipped gates for runtime model/application behavior when
  dependencies are unavailable and for SHAP behavior while the artifact is
  absent.

The integration test uses a patched in-memory `LoadedModel` and fake pipeline,
not a production artifact. No SHAP implementation or substitute feature
attribution is provided.
