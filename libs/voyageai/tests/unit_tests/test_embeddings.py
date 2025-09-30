"""Test embedding model integration."""

from typing import List

from langchain_core.embeddings import Embeddings
from pydantic import SecretStr

from langchain_voyageai import VoyageAIEmbeddings

MODEL = "voyage-2"


def test_initialization_voyage_2() -> None:
    """Test embedding model initialization."""
    emb = VoyageAIEmbeddings(voyage_api_key=SecretStr("NOT_A_VALID_KEY"), model=MODEL)  # type: ignore
    assert isinstance(emb, Embeddings)
    assert emb.batch_size == 72
    assert emb.model == MODEL
    assert emb._client is not None


def test_initialization_voyage_2_with_full_api_key_name() -> None:
    """Test embedding model initialization."""
    # Testing that we can initialize the model using `voyage_api_key`
    # instead of `api_key`
    emb = VoyageAIEmbeddings(voyage_api_key=SecretStr("NOT_A_VALID_KEY"), model=MODEL)  # type: ignore
    assert isinstance(emb, Embeddings)
    assert emb.batch_size == 72
    assert emb.model == MODEL
    assert emb._client is not None


def test_initialization_voyage_1() -> None:
    """Test embedding model initialisation."""
    emb = VoyageAIEmbeddings(
        voyage_api_key=SecretStr("NOT_A_VALID_KEY"), model="voyage-01"
    )  # type: ignore
    assert isinstance(emb, Embeddings)
    assert emb.batch_size == 7
    assert emb.model == "voyage-01"
    assert emb._client is not None


def test_initialization_voyage_1_batch_size() -> None:
    """Test embedding model initialization."""
    emb = VoyageAIEmbeddings(
        voyage_api_key=SecretStr("NOT_A_VALID_KEY"),  # type: ignore
        model="voyage-01",
        batch_size=15,
    )
    assert isinstance(emb, Embeddings)
    assert emb.batch_size == 15
    assert emb.model == "voyage-01"
    assert emb._client is not None


def test_initialization_with_output_dimension() -> None:
    emb = VoyageAIEmbeddings(
        voyage_api_key=SecretStr("NOT_A_VALID_KEY"),  # type: ignore
        model="voyage-3-large",
        output_dimension=256,
        batch_size=10,
    )
    assert isinstance(emb, Embeddings)
    assert emb.model == "voyage-3-large"
    assert emb.output_dimension == 256


def test_initialization_contextual_model() -> None:
    """Test initialization with contextual embedding model."""
    emb = VoyageAIEmbeddings(
        voyage_api_key=SecretStr("NOT_A_VALID_KEY"), model="voyage-context-3"
    )  # type: ignore
    assert isinstance(emb, Embeddings)
    assert emb.model == "voyage-context-3"
    assert emb.batch_size == 7  # Default batch size for contextual models
    assert emb._client is not None


def test_initialization_contextual_model_with_custom_batch_size() -> None:
    """Test initialization of contextual model with custom batch size."""
    emb = VoyageAIEmbeddings(
        voyage_api_key=SecretStr("NOT_A_VALID_KEY"),  # type: ignore
        model="voyage-context-3",
        batch_size=5,
    )
    assert isinstance(emb, Embeddings)
    assert emb.model == "voyage-context-3"
    assert emb.batch_size == 5
    assert emb._client is not None


def test_initialization_contextual_model_with_output_dimension() -> None:
    """Test initialization of contextual model with output dimension."""
    emb = VoyageAIEmbeddings(
        voyage_api_key=SecretStr("NOT_A_VALID_KEY"),  # type: ignore
        model="voyage-context-3",
        output_dimension=512,
    )
    assert isinstance(emb, Embeddings)
    assert emb.model == "voyage-context-3"
    assert emb.output_dimension == 512
    assert emb._client is not None


def test_is_context_model_detection() -> None:
    """Test contextual model detection."""
    # Contextual model
    emb_context = VoyageAIEmbeddings(
        voyage_api_key=SecretStr("NOT_A_VALID_KEY"), model="voyage-context-3"
    )  # type: ignore
    assert emb_context._is_context_model() is True

    # Regular model
    emb_regular = VoyageAIEmbeddings(
        voyage_api_key=SecretStr("NOT_A_VALID_KEY"), model="voyage-3"
    )  # type: ignore
    assert emb_regular._is_context_model() is False

    # Another regular model
    emb_regular2 = VoyageAIEmbeddings(
        voyage_api_key=SecretStr("NOT_A_VALID_KEY"), model="voyage-2"
    )  # type: ignore
    assert emb_regular2._is_context_model() is False


def test_contextual_model_variants() -> None:
    """Test different contextual model variants."""
    context_models = [
        "voyage-context-3",
        "voyage-context-lite",
        "custom-context-model",
    ]

    for model in context_models:
        emb = VoyageAIEmbeddings(
            voyage_api_key=SecretStr("NOT_A_VALID_KEY"), model=model
        )  # type: ignore
        assert (
            emb._is_context_model() is True
        ), f"Model {model} should be detected as contextual"


class _StubResponse:
    def __init__(self, count: int) -> None:
        self.embeddings = [[float(i)] for i in range(count)]


class _StubClient:
    def __init__(self, token_lengths: List[int], recorded_batches: List[List[str]]) -> None:
        self._token_lengths = token_lengths
        self._recorded_batches = recorded_batches

    def tokenize(self, texts: List[str], model: str) -> List[List[int]]:  # type: ignore[override]
        assert len(texts) == len(self._token_lengths)
        return [list(range(length)) for length in self._token_lengths]

    def embed(self, texts: List[str], **_: object) -> _StubResponse:  # type: ignore[override]
        batch = list(texts)
        self._recorded_batches.append(batch)
        return _StubResponse(len(batch))


def test_embed_regular_splits_on_token_limit(monkeypatch) -> None:
    texts = ["text-a", "text-b", "text-c", "text-d"]
    # voyage-3.5 limit is 320k tokens per request. Force batches of two items each.
    token_lengths = [150_000, 150_000, 150_000, 150_000]
    recorded_batches: List[List[str]] = []
    emb = VoyageAIEmbeddings(
        voyage_api_key=SecretStr("NOT_A_VALID_KEY"),  # type: ignore
        model="voyage-3.5",
        batch_size=10,
    )
    stub_client = _StubClient(token_lengths, recorded_batches)
    monkeypatch.setattr(emb, "_client", stub_client, raising=False)

    result = emb._embed_regular(texts, "document")

    assert recorded_batches == [["text-a", "text-b"], ["text-c", "text-d"]]
    assert len(result) == len(texts)


def test_iter_token_safe_batch_respects_custom_batch_size(monkeypatch) -> None:
    texts = [f"chunk-{i}" for i in range(5)]
    token_lengths = [5] * len(texts)
    recorded_batches: List[List[str]] = []
    emb = VoyageAIEmbeddings(
        voyage_api_key=SecretStr("NOT_A_VALID_KEY"),  # type: ignore
        model="voyage-3.5-lite",
        batch_size=2,
    )
    stub_client = _StubClient(token_lengths, recorded_batches)
    monkeypatch.setattr(emb, "_client", stub_client, raising=False)

    slices = list(emb._iter_token_safe_batch_slices(texts))
    assert slices == [(0, 2), (2, 4), (4, 5)]


def test_iter_token_safe_batch_handles_single_oversized_text(monkeypatch) -> None:
    texts = ["oversized"]
    token_lengths = [500_000]
    recorded_batches: List[List[str]] = []
    emb = VoyageAIEmbeddings(
        voyage_api_key=SecretStr("NOT_A_VALID_KEY"),  # type: ignore
        model="voyage-3-large",
        batch_size=5,
    )
    stub_client = _StubClient(token_lengths, recorded_batches)
    monkeypatch.setattr(emb, "_client", stub_client, raising=False)

    slices = list(emb._iter_token_safe_batch_slices(texts))
    assert slices == [(0, 1)]
