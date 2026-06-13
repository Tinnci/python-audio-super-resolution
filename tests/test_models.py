from audio_super_resolution import list_models


def test_list_models_includes_baseline_and_audiosr_models() -> None:
    models = list_models()
    model_ids = {model.id for model in models}

    assert model_ids == {"sinc-resample", "audiosr-basic", "audiosr-speech"}
    assert any(model.backend == "sinc-resample" and model.target_sample_rate is None for model in models)
    assert any(model.backend == "audiosr" and model.target_sample_rate == 48000 for model in models)


def test_list_models_filters_by_backend_or_name() -> None:
    assert {model.id for model in list_models("speech")} == {"audiosr-speech"}
    assert {model.id for model in list_models("sinc")} == {"sinc-resample"}
