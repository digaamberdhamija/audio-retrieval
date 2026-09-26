from app.core.models import SpeakerWord, Window


def load_words(transcript: dict) -> list[SpeakerWord]:
    return [
        SpeakerWord(segment["speaker"], word["text"], word["start"], word["end"])
        for segment in transcript["segments"]
        for word in segment["words"]
        if word["start"] is not None and word["end"] is not None
    ]


def render_text(words: list[SpeakerWord]) -> str:
    lines = []
    current_speaker = None
    for word in words:
        if word.speaker != current_speaker:
            lines.append(f"{word.speaker}:")
            current_speaker = word.speaker
        lines[-1] += f" {word.text}"
    return "\n".join(lines)


def speakers_in(words: list[SpeakerWord]) -> list[str]:
    return list(dict.fromkeys(word.speaker for word in words))


def words_between(words: list[SpeakerWord], start: float, end: float) -> list[SpeakerWord]:
    return [word for word in words if start <= word.start and word.end <= end]


def sliding_windows(words: list[SpeakerWord], length_seconds: int, stride_seconds: float) -> list[Window]:
    duration = max(word.end for word in words)
    windows = []
    window_start = 0.0
    while True:
        selected = [w for w in words if window_start <= w.start < window_start + length_seconds]
        if selected:
            windows.append(
                Window(
                    window_seconds=length_seconds,
                    start_seconds=min(word.start for word in selected),
                    end_seconds=max(word.end for word in selected),
                    speakers=speakers_in(selected),
                    text=render_text(selected),
                )
            )
        if window_start + length_seconds >= duration:
            break
        window_start += stride_seconds
    return windows
