from exceptions import (
    ArtifactError,
    DockerError,
    InferEdgeError,
    ProbeTimeoutError,
    TransientArtifactError,
    TransientDockerError,
    VllmLoadError,
)


def test_exception_hierarchy():
    assert issubclass(ArtifactError, InferEdgeError)
    assert issubclass(TransientArtifactError, ArtifactError)
    assert issubclass(DockerError, InferEdgeError)
    assert issubclass(TransientDockerError, DockerError)
    assert issubclass(ProbeTimeoutError, InferEdgeError)
    assert issubclass(VllmLoadError, InferEdgeError)


def test_exceptions_are_catchable_by_base():
    for exc_cls in (
        ArtifactError,
        TransientArtifactError,
        DockerError,
        TransientDockerError,
        ProbeTimeoutError,
        VllmLoadError,
    ):
        try:
            raise exc_cls("test")
        except InferEdgeError as exc:
            assert "test" in str(exc)