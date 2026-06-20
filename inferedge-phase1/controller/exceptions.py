class InferEdgeError(Exception):
    """Base exception for InferEdge controller."""


class ArtifactError(InferEdgeError):
    """Model artifact download or cache failure (permanent — operator action likely needed)."""


class TransientArtifactError(ArtifactError):
    """Recoverable download/network issue — reconciler will retry automatically."""


class DockerError(InferEdgeError):
    """Docker daemon or container operation failure."""


class TransientDockerError(DockerError):
    """Recoverable Docker issue — reconciler will auto-heal and retry."""


class ProbeTimeoutError(InferEdgeError):
    """vLLM health/model probe timed out."""


class VllmLoadError(InferEdgeError):
    """vLLM failed to load the model (container crash, CUDA/driver mismatch, etc.)."""