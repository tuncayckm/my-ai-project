from pydantic import BaseModel
from typing import Optional, Dict

class ApplyKnowledgeRequest(BaseModel):
    user_id: str
    prompt: str
    token: str
    plugin_data: Optional[dict] = None
    max_context_tokens: Optional[int] = None
