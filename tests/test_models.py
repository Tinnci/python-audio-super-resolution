from audio_super_resolution import list_models
from audio_super_resolution.backends import list_backend_model_specs, registered_backend_types


def test_list_models_includes_baseline_and_audiosr_models() -> None:
    models = list_models()
    model_ids = {model.id for model in models}

    assert model_ids == {"sinc-resample", "audiosr-basic", "audiosr-speech", "lavasr-v2-bwe"}
    assert any(model.backend == "sinc-resample" and model.target_sample_rate is None for model in models)
    assert any(model.backend == "audiosr" and model.target_sample_rate == 48000 for model in models)
    assert any(model.backend == "lavasr-compat" and model.requires_weights for model in models)


def test_list_models_filters_by_backend_or_name() -> None:
    assert {model.id for model in list_models("speech")} == {"audiosr-speech"}
    assert {model.id for model in list_models("sinc")} == {"sinc-resample"}


def test_list_models_exposes_backend_metadata() -> None:
    models = {model.id: model for model in list_models()}

    assert models["sinc-resample"].implementation == "baseline"
    assert models["sinc-resample"].domain == ("general",)
    assert models["sinc-resample"].fixed_target_sr is False
    assert models["audiosr-basic"].implementation == "external_package"
    assert models["audiosr-basic"].target_sample_rates == (48000,)
    assert models["audiosr-basic"].fixed_target_sr is True
    assert models["lavasr-v2-bwe"].implementation == "self_torch"
    assert models["lavasr-v2-bwe"].weight_provider == "huggingface"
    assert models["lavasr-v2-bwe"].weights_license == "Apache-2.0"


def test_backend_registry_exposes_builtin_model_specs() -> None:
    backends = registered_backend_types()
    specs = {spec.id for spec in list_backend_model_specs()}

    assert {"audiosr", "sinc-resample", "lavasr-compat"} <= set(backends)
    assert {"audiosr-basic", "audiosr-speech", "sinc-resample", "lavasr-v2-bwe"} == specs
