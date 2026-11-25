from pydantic import BaseModel, Field


class Document(BaseModel):
    """Текст документа и метаданные его источника."""

    document_id: str = Field(min_length=1)
    source_name: str = Field(min_length=1)
    text: str = Field(min_length=1)


class Chunk(BaseModel):
    """Фрагмент исходного документа с метаданными."""

    document_id: str = Field(min_length=1)
    source_name: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
