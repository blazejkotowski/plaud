from ddsp.interfaces import ControlField, ControlSpace


def test_controlspace_dims_and_names():
    cs = ControlSpace(tuple([
        ControlField(name="loudness", dim=1, source="feature"),
        ControlField(name="centroid", dim=1, source="feature"),
        ControlField(name="latents", dim=4, source="latent"),
    ]))

    assert cs.feature_dim == 2
    assert cs.latent_dim == 4
    assert cs.total_dim == 6
    assert cs.names() == ("loudness", "centroid", "latents")


def test_controlspace_empty():
    cs = ControlSpace(tuple([]))
    assert cs.feature_dim == 0
    assert cs.latent_dim == 0
    assert cs.total_dim == 0
    assert cs.names() == tuple()
