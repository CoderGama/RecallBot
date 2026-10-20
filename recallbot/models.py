from dataclasses import dataclass


@dataclass(frozen=True)
class Fact:
    attribute: str
    value: str
    statement: str
    subject: str = "user"


@dataclass(frozen=True)
class Reply:
    text: str
    event: str = "message"
