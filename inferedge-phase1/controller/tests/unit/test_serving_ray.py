import sys
from unittest.mock import MagicMock, patch

from schemas import DesiredState
from serving.ray_cluster import RayClusterServingBackend, _app_name


def test_app_name_from_config_hash():
    assert _app_name("abc123def456") == "inferedge-abc123def456"


def test_build_llm_config_maps_desired_state(tmp_path):
    from serving.ray_cluster import _build_llm_config

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"quantization_config": {"quant_method": "awq"}}')

    desired = DesiredState(model="org/model", context_length=4096, gpu_utilization=0.9)
    mock_llm_config = MagicMock()
    mock_module = MagicMock()
    mock_module.LLMConfig = mock_llm_config

    with patch.dict(sys.modules, {"ray": MagicMock(), "ray.serve": MagicMock(), "ray.serve.llm": mock_module}):
        _build_llm_config("org/model", str(model_dir), desired)

    mock_llm_config.assert_called_once()
    kwargs = mock_llm_config.call_args.kwargs
    assert kwargs["model_loading_config"]["model_id"] == "org/model"
    assert kwargs["model_loading_config"]["model_source"] == str(model_dir)
    assert kwargs["engine_kwargs"]["max_model_len"] == 4096
    assert kwargs["engine_kwargs"]["gpu_memory_utilization"] == 0.9
    assert kwargs["engine_kwargs"]["quantization"] == "awq"


def test_ray_backend_mode():
    scheduler = MagicMock()
    backend = RayClusterServingBackend(scheduler)
    assert backend.mode == "ray_cluster"