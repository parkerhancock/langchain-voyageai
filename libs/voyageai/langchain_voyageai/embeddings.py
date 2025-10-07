import logging
from typing import Any, Iterable, Iterator, List, Literal, Optional, Tuple, cast

import voyageai  # type: ignore
from langchain_core.embeddings import Embeddings
from langchain_core.utils import secret_from_env
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SecretStr,
    model_validator,
)
from typing_extensions import Self

logger = logging.getLogger(__name__)

DEFAULT_VOYAGE_2_BATCH_SIZE = 72
DEFAULT_VOYAGE_3_LITE_BATCH_SIZE = 30
DEFAULT_VOYAGE_3_BATCH_SIZE = 10
DEFAULT_BATCH_SIZE = 7
MAX_DOCUMENTS_PER_REQUEST = 1_000
DEFAULT_MAX_TOKENS_PER_REQUEST = 120_000
TOKEN_LIMIT_OVERRIDES: Tuple[Tuple[int, Tuple[str, ...]], ...] = (
    (1_000_000, ("voyage-3.5-lite", "voyage-3-lite")),
    (320_000, ("voyage-3.5", "voyage-3", "voyage-2", "voyage-02")),
)


class VoyageAIEmbeddings(BaseModel, Embeddings):
    """VoyageAIEmbeddings embedding model.

    Example:
        .. code-block:: python

            from langchain_voyageai import VoyageAIEmbeddings

            model = VoyageAIEmbeddings()
    """

    _client: voyageai.Client = PrivateAttr()
    _aclient: voyageai.client_async.AsyncClient = PrivateAttr()
    model: str
    batch_size: int

    output_dimension: Optional[Literal[256, 512, 1024, 2048]] = None
    show_progress_bar: bool = False
    truncation: bool = True
    voyage_api_key: SecretStr = Field(
        alias="api_key",
        default_factory=secret_from_env(
            "VOYAGE_API_KEY",
            error_message="Must set `VOYAGE_API_KEY` environment variable or "
            "pass `api_key` to VoyageAIEmbeddings constructor.",
        ),
    )

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )

    @model_validator(mode="before")
    @classmethod
    def default_values(cls, values: dict) -> Any:
        """Set default batch size based on model"""
        model = values.get("model")
        batch_size = values.get("batch_size")
        if batch_size is None:
            values["batch_size"] = (
                DEFAULT_VOYAGE_2_BATCH_SIZE
                if model in ["voyage-2", "voyage-02"]
                else (
                    DEFAULT_VOYAGE_3_LITE_BATCH_SIZE
                    if model in ["voyage-3-lite", "voyage-3.5-lite"]
                    else (
                        DEFAULT_VOYAGE_3_BATCH_SIZE
                        if model in ["voyage-3", "voyage-3.5"]
                        else DEFAULT_BATCH_SIZE
                    )
                )
            )
        return values

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that VoyageAI credentials exist in environment."""
        api_key_str = self.voyage_api_key.get_secret_value()
        self._client = voyageai.Client(api_key=api_key_str)
        self._aclient = voyageai.client_async.AsyncClient(api_key=api_key_str)
        return self

    def _max_documents_per_batch(self) -> int:
        """Return the maximum number of documents allowed in a single request."""
        return max(1, min(self.batch_size, MAX_DOCUMENTS_PER_REQUEST))

    def _max_tokens_per_batch(self) -> int:
        """Return the maximum number of tokens allowed for the current model."""
        model_name = self.model
        for limit, models in TOKEN_LIMIT_OVERRIDES:
            if model_name in models:
                return limit
        return DEFAULT_MAX_TOKENS_PER_REQUEST

    def _token_lengths(self, texts: List[str]) -> List[int]:
        """Return token lengths for texts using the Voyage client tokenizer."""
        try:
            tokenized = self._client.tokenize(texts, self.model)
        except Exception:
            logger.debug("Failed to tokenize texts for model %s", self.model)
            raise
        return [len(tokens) for tokens in tokenized]

    def _iter_token_safe_batch_slices(
        self, texts: List[str]
    ) -> Iterator[Tuple[int, int]]:
        """Yield (start, end) indices for batches within token and length limits."""
        if not texts:
            return

        token_lengths = self._token_lengths(texts)
        max_docs = self._max_documents_per_batch()
        max_tokens = self._max_tokens_per_batch()

        index = 0
        total_texts = len(texts)
        while index < total_texts:
            start = index
            batch_tokens = 0
            batch_docs = 0
            while index < total_texts and batch_docs < max_docs:
                current_tokens = token_lengths[index]
                if batch_docs > 0 and batch_tokens + current_tokens > max_tokens:
                    break

                if current_tokens > max_tokens and batch_docs == 0:
                    logger.warning(
                        "Text at index %s exceeds Voyage token limit (%s > %s). "
                        "Sending as a single-item batch; API may truncate or error.",
                        index,
                        current_tokens,
                        max_tokens,
                    )
                    index += 1
                    batch_docs += 1
                    batch_tokens = current_tokens
                    break

                batch_tokens += current_tokens
                batch_docs += 1
                index += 1

            if start == index:
                index += 1
            yield (start, index)

    def _is_context_model(self) -> bool:
        """Check if the model is a contextualized embedding model."""
        return "context" in self.model

    def _embed_context(
        self, inputs: List[List[str]], input_type: str
    ) -> List[List[float]]:
        """Embed using contextualized embedding API."""
        r = self._client.contextualized_embed(
            inputs=inputs,
            model=self.model,
            input_type=input_type,
            output_dimension=self.output_dimension,
        ).results
        return r[0].embeddings  # type: ignore

    def _embed_regular(self, texts: List[str], input_type: str) -> List[List[float]]:
        """Embed using regular embedding API."""
        embeddings: List[List[float]] = []
        progress = None
        if self.show_progress_bar:
            try:
                from tqdm.auto import tqdm  # type: ignore
            except ImportError as e:
                raise ImportError(
                    "Must have tqdm installed if `show_progress_bar` is set to True. "
                    "Please install with `pip install tqdm`."
                ) from e

            progress = tqdm(total=len(texts))

        try:
            for start, end in self._iter_token_safe_batch_slices(texts):
                if start == end:
                    continue
                batch = texts[start:end]
                r = self._client.embed(
                    batch,
                    model=self.model,
                    input_type=input_type,
                    truncation=self.truncation,
                    output_dimension=self.output_dimension,
                ).embeddings
                embeddings.extend(cast(Iterable[List[float]], r))
                if progress is not None:
                    progress.update(len(batch))
        finally:
            if progress is not None:
                progress.close()
        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        if self._is_context_model():
            return self._embed_context([texts], "document")
        return self._embed_regular(texts, "document")

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        if self._is_context_model():
            result = self._embed_context([[text]], "query")
        else:
            result = self._embed_regular([text], "query")
        return result[0]

    async def _aembed_context(
        self, inputs: List[List[str]], input_type: str
    ) -> List[List[float]]:
        """Async embed using contextualized embedding API."""
        r = await self._aclient.contextualized_embed(
            inputs=inputs,
            model=self.model,
            input_type=input_type,
            output_dimension=self.output_dimension,
        )
        return r.results[0].embeddings  # type: ignore

    async def _aembed_regular(
        self, texts: List[str], input_type: str
    ) -> List[List[float]]:
        """Async embed using regular embedding API."""
        embeddings: List[List[float]] = []
        progress = None
        if self.show_progress_bar:
            try:
                from tqdm.auto import tqdm  # type: ignore
            except ImportError as e:
                raise ImportError(
                    "Must have tqdm installed if `show_progress_bar` is set to True. "
                    "Please install with `pip install tqdm`."
                ) from e

            progress = tqdm(total=len(texts))

        try:
            for start, end in self._iter_token_safe_batch_slices(texts):
                if start == end:
                    continue
                batch = texts[start:end]
                r = await self._aclient.embed(
                    batch,
                    model=self.model,
                    input_type=input_type,
                    truncation=self.truncation,
                    output_dimension=self.output_dimension,
                )
                embeddings.extend(cast(Iterable[List[float]], r.embeddings))
                if progress is not None:
                    progress.update(len(batch))
        finally:
            if progress is not None:
                progress.close()
        return embeddings

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async embed search docs."""
        if self._is_context_model():
            return await self._aembed_context([texts], "document")
        return await self._aembed_regular(texts, "document")

    async def aembed_query(self, text: str) -> List[float]:
        """Async embed query text."""
        if self._is_context_model():
            result = await self._aembed_context([[text]], "query")
        else:
            result = await self._aembed_regular([text], "query")
        return result[0]
