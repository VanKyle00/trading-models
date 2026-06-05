from tradinglib.training.config import LoRASettings, TrainSettings


def test_lora_defaults():
    lora = LoRASettings()
    assert lora.r == 16 and lora.alpha == 32
    # Qwen2 attention + MLP projections
    assert "q_proj" in lora.target_modules and "down_proj" in lora.target_modules


def test_train_defaults():
    t = TrainSettings()
    assert t.base_model.startswith("unsloth/") or "Qwen2.5-7B" in t.base_model
    assert t.max_seq_len == 2048
    assert 0 < t.learning_rate < 1
    assert t.load_in_4bit is True


def test_overrides_apply():
    t = TrainSettings(epochs=3, learning_rate=1e-4)
    assert t.epochs == 3 and t.learning_rate == 1e-4


def test_settings_are_frozen():
    t = TrainSettings()
    import dataclasses

    try:
        t.epochs = 9  # type: ignore[misc]
        raised = False
    except dataclasses.FrozenInstanceError:
        raised = True
    assert raised
