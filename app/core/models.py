from dataclasses import dataclass, field


@dataclass
class Word:
    text: str
    start: float | None
    end: float | None


@dataclass
class Segment:
    speaker: str
    start: float | None
    end: float | None
    text: str
    words: list[Word] = field(default_factory=list)


@dataclass
class Transcript:
    audio_file: str
    model: str
    speakers: dict[str, str]
    segments: list[Segment]


@dataclass
class SpeakerWord:
    speaker: str
    text: str
    start: float
    end: float


@dataclass
class Window:
    window_seconds: int
    start_seconds: float
    end_seconds: float
    speakers: list[str]
    text: str


@dataclass
class Chunk:
    chunk_ids: list[int]
    transcript_id: int
    audio_file: str
    window_seconds: int
    start_seconds: float
    end_seconds: float
    speakers: list[str]
    text: str
    score: float = 0.0
    rerank_score: float = 0.0
