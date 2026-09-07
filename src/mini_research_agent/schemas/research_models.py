from pydantic import BaseModel,Field



class Summary(BaseModel):
    key_excerpts: str = Field(description="Important quotes and excerpts from the content")
    summary: str = Field(description="Concise summary of the webpage content")