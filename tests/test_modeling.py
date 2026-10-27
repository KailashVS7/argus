from argus.modeling import build_model


def test_model_construction_no_download_offline_safe():
    model, meta = build_model(weights="imagenet", no_download=True)
    assert model is not None
    assert meta.requested_weights == "imagenet"
    assert meta.no_download is True
    # either cached ImageNet or safe fallback to random init
    assert isinstance(meta.used_imagenet_weights, bool)
