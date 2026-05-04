from unittest.mock import MagicMock, patch
import pytest
from app.core.model.local_embedding import LocalEmbeddingProvider
from app.core.model.factory import get_embedder
from app.config.settings import AppSettings, LocalEmbeddingConfig, ProviderConfig


@pytest.fixture
def mock_st(tmp_path):
    """Patch SentenceTransformer to avoid real model loading."""
    with patch("app.core.model.local_embedding.SentenceTransformer") as MockST:
        mock_model = MagicMock()
        MockST.return_value = mock_model
        yield mock_model, tmp_path


def test_embed_returns_list_of_lists(mock_st):
    import numpy as np
    mock_model, _ = mock_st
    mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
    provider = LocalEmbeddingProvider("/fake/path")
    result = provider.embed(["a", "b"])
    assert len(result) == 2
    assert isinstance(result[0], list)


def test_chat_raises(mock_st):
    mock_model, _ = mock_st
    provider = LocalEmbeddingProvider("/fake/path")
    with pytest.raises(NotImplementedError):
        provider.chat([])


def test_health_check_true(mock_st):
    import numpy as np
    mock_model, _ = mock_st
    mock_model.encode.return_value = np.array([[0.1]])
    provider = LocalEmbeddingProvider("/fake/path")
    assert provider.health_check() is True


def test_get_embedder_uses_local_when_enabled(tmp_path):
    """get_embedder returns LocalEmbeddingProvider when enabled=True and path exists."""
    model_dir = tmp_path / "mymodel"
    model_dir.mkdir()
    settings = AppSettings(
        local_embedding=LocalEmbeddingConfig(enabled=True, model_path=str(model_dir)),
        providers=[ProviderConfig(name="x", api_key="k", base_url="", chat_model="m", embed_model="e")],
        active_provider="x",
        data_dir=str(tmp_path),
    )
    with patch("app.core.model.local_embedding.SentenceTransformer") as MockST:
        MockST.return_value = MagicMock()
        from app.core.model.local_embedding import LocalEmbeddingProvider
        result = get_embedder(settings)
        assert isinstance(result, LocalEmbeddingProvider)
