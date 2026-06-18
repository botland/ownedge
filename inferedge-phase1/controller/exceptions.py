class InferEdgeError(Exception):
    """Base exception for InferEdge controller."""


class ArtifactError(InferEdgeError):
    """Model artifact download or cache failure."""


class DockerError(InferEdgeError):
    """Docker daemon or container operation failure."""


class ProbeTimeoutError(InferEdgeError):
    """vLLM health/model probe timed out."""