"""用于嵌入模型前处理的递归文本分块工具。"""

from collections.abc import Sequence
from dataclasses import dataclass, field


DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 64
DEFAULT_SEPARATORS = (
    "\n\n",
    "\n",
    "。",
    "！",
    "？",
    ";",
    "；",
    ",",
    "，",
    " ",
    "",
)


@dataclass(frozen=True, slots=True)
class RecursiveTextSplitter:
    """按分隔符优先级递归拆分文本，并合并为可控大小的文本块。

    ``chunk_size`` 和 ``chunk_overlap`` 均以 Python 字符数计算。拆分时优先
    保留段落和句子边界；某段仍然过长时，会继续使用下一级分隔符，最后逐字符
    拆分，保证每个结果都不超过 ``chunk_size``。
    """

    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    separators: Sequence[str] = field(default_factory=lambda: DEFAULT_SEPARATORS)
    strip_whitespace: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.chunk_size, bool) or not isinstance(self.chunk_size, int):
            raise TypeError("chunk_size must be an integer")
        if isinstance(self.chunk_overlap, bool) or not isinstance(
            self.chunk_overlap, int
        ):
            raise TypeError("chunk_overlap must be an integer")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be greater than or equal to 0")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        separators = tuple(self.separators)
        if not separators:
            separators = ("",)
        if any(not isinstance(separator, str) for separator in separators):
            raise TypeError("separators must contain only strings")
        if "" not in separators:
            separators += ("",)

        object.__setattr__(self, "separators", separators)

    def split(self, text: str) -> list[str]:
        """将一段文本拆成适合批量生成嵌入向量的文本块。"""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not text or (self.strip_whitespace and not text.strip()):
            return []
        if len(text) <= self.chunk_size:
            chunk = self._normalize(text)
            return [chunk] if chunk else []

        return self._split_recursive(text, tuple(self.separators))

    def _split_recursive(self, text: str, separators: tuple[str, ...]) -> list[str]:
        separator_index = self._find_separator(text, separators)
        separator = separators[separator_index]
        remaining_separators = separators[separator_index + 1 :]
        pieces = self._split_preserving_separator(text, separator)

        chunks: list[str] = []
        mergeable_pieces: list[str] = []

        for piece in pieces:
            if len(piece) <= self.chunk_size:
                mergeable_pieces.append(piece)
                continue

            chunks.extend(self._merge(mergeable_pieces))
            mergeable_pieces = []
            chunks.extend(self._split_recursive(piece, remaining_separators))

        chunks.extend(self._merge(mergeable_pieces))
        return chunks

    @staticmethod
    def _find_separator(text: str, separators: tuple[str, ...]) -> int:
        for index, separator in enumerate(separators):
            if not separator or separator in text:
                return index
        return len(separators) - 1

    @staticmethod
    def _split_preserving_separator(text: str, separator: str) -> list[str]:
        if not separator:
            return list(text)

        raw_pieces = text.split(separator)
        pieces = [piece + separator for piece in raw_pieces[:-1]]
        if raw_pieces[-1]:
            pieces.append(raw_pieces[-1])
        return [piece for piece in pieces if piece]

    def _merge(self, pieces: list[str]) -> list[str]:
        chunks: list[str] = []
        current_pieces: list[str] = []
        current_size = 0

        for piece in pieces:
            piece_size = len(piece)
            if current_pieces and current_size + piece_size > self.chunk_size:
                self._append_chunk(chunks, current_pieces)

                while current_pieces and current_size > self.chunk_overlap:
                    current_size -= len(current_pieces.pop(0))
                while (
                    current_pieces
                    and current_size + piece_size > self.chunk_size
                ):
                    current_size -= len(current_pieces.pop(0))

            current_pieces.append(piece)
            current_size += piece_size

        self._append_chunk(chunks, current_pieces)
        return chunks

    def _append_chunk(self, chunks: list[str], pieces: list[str]) -> None:
        chunk = self._normalize("".join(pieces))
        if chunk:
            chunks.append(chunk)

    def _normalize(self, text: str) -> str:
        return text.strip() if self.strip_whitespace else text


def split_text(
    text: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    separators: Sequence[str] = DEFAULT_SEPARATORS,
    strip_whitespace: bool = True,
) -> list[str]:
    """使用一次性配置递归拆分文本。"""
    return RecursiveTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        strip_whitespace=strip_whitespace,
    ).split(text)
