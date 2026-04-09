"""
Universal LLM adapter for HealthLink.
Single function interface using LangChain 1.x with Google GenAI integration.
Supports Gemini models with structured outputs via Pydantic.

Updated for:
- LangChain 1.x (requires Python 3.10+)
- langchain-google-genai 3.x
"""
import json
import logging
from typing import Type, TypeVar, Optional, Dict, Any
from pydantic import BaseModel, ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential

# LangChain 1.x imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from config.settings import Settings


logger = logging.getLogger("healthlink.llm")

T = TypeVar('T', bound=BaseModel)


class LLMClient:
    """LLM client wrapper using LangChain Google GenAI (v3.x)."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model_name = settings.llm_model_name

        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=settings.gemini_api_key,
            temperature=settings.llm_temperature,
            max_output_tokens=settings.llm_max_tokens,
            max_retries=2,
        )

        logger.info(f"LLM client initialized with model: {self.model_name}")

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_instruction: Optional[str] = None
    ) -> str:
        """
        Generate text using Gemini model.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature (uses settings default if None)
            max_tokens: Maximum tokens (uses settings default if None)
            system_instruction: Optional system instruction

        Returns:
            Generated text response
        """
        messages = []

        if system_instruction:
            messages.append(SystemMessage(content=system_instruction))

        messages.append(HumanMessage(content=prompt))

        llm = self.llm
        if temperature is not None or max_tokens is not None:
            llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=self.settings.gemini_api_key,
                temperature=temperature if temperature is not None else self.settings.llm_temperature,
                max_output_tokens=max_tokens if max_tokens is not None else self.settings.llm_max_tokens,
                max_retries=2,
            )

        response: AIMessage = llm.invoke(messages)

        return response.content

    def generate_structured(
        self,
        prompt: str,
        response_schema: Type[T],
        temperature: Optional[float] = None,
        system_instruction: Optional[str] = None
    ) -> T:
        """
        Generate structured output using LangChain's with_structured_output.

        Args:
            prompt: Input prompt
            response_schema: Pydantic model for response structure
            temperature: Sampling temperature
            system_instruction: Optional system instruction

        Returns:
            Validated Pydantic model instance
        """
        messages = []

        if system_instruction:
            messages.append(SystemMessage(content=system_instruction))

        messages.append(HumanMessage(content=prompt))

        llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self.settings.gemini_api_key,
            temperature=temperature if temperature is not None else self.settings.llm_temperature,
            max_retries=2,
        )

        structured_llm = llm.with_structured_output(response_schema)

        return structured_llm.invoke(messages)


_llm_client: Optional[LLMClient] = None


def get_llm_client(settings: Settings) -> LLMClient:
    """
    FastAPI dependency for LLM client.

    Args:
        settings: Application settings

    Returns:
        LLM client instance
    """
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient(settings)
    return _llm_client


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def llm_generate(
    prompt: str,
    schema: Type[T],
    temperature: Optional[float] = None,
    context: Optional[str] = None,
    client: Optional[LLMClient] = None,
) -> T:
    """
    Generate structured output from LLM with automatic validation and retry.

    This is the SINGLE function used by all agents for LLM calls.
    Uses LangChain's with_structured_output for native JSON support.

    Args:
        prompt: The prompt to send to the LLM
        schema: Pydantic model class for structured output
        temperature: Sampling temperature (optional)
        context: Additional context to include (optional)
        client: LLM client instance (uses global if None)

    Returns:
        Validated Pydantic model instance

    Raises:
        ValidationError: If LLM output cannot be validated after retries
        Exception: If LLM generation fails
    """
    if client is None:
        from config.settings import get_settings
        settings = get_settings()
        client = get_llm_client(settings)

    full_prompt = f"""TASK:
{prompt}
"""

    if context:
        full_prompt += f"\n\nCONTEXT:\n{context}"

    logger.debug(f"Generating with schema: {schema.__name__}")

    try:
        result = client.generate_structured(
            prompt=full_prompt,
            response_schema=schema,
            temperature=temperature,
            system_instruction="You are a helpful medical information assistant. Provide structured, accurate responses."
        )

        logger.info(f"Successfully generated and validated {schema.__name__}")
        return result

    except Exception as e:
        logger.warning(f"Structured generation failed, falling back to text mode: {e}")
        return generate_with_text_fallback(client, prompt, schema, temperature, context)


def generate_with_text_fallback(
    client: LLMClient,
    prompt: str,
    schema: Type[T],
    temperature: Optional[float],
    context: Optional[str]
) -> T:
    """
    Fallback method using text generation with JSON parsing.
    Used when structured output fails.
    """
    schema_json = schema.model_json_schema()
    schema_description = json.dumps(schema_json, indent=2)

    enhanced_prompt = f"""You are a medical information assistant. Respond with ONLY valid JSON matching the schema below.

SCHEMA:
{schema_description}

TASK:
{prompt}
"""

    if context:
        enhanced_prompt += f"\n\nCONTEXT:\n{context}"

    enhanced_prompt += "\n\nRESPONSE (JSON only, no markdown, no explanation):"

    response_text = client.generate(prompt=enhanced_prompt, temperature=temperature)

    cleaned_response = response_text.strip()
    if cleaned_response.startswith("```json"):
        cleaned_response = cleaned_response[7:]
    if cleaned_response.startswith("```"):
        cleaned_response = cleaned_response[3:]
    if cleaned_response.endswith("```"):
        cleaned_response = cleaned_response[:-3]
    cleaned_response = cleaned_response.strip()

    try:
        response_dict = json.loads(cleaned_response)
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}. Response: {cleaned_response}")
        raise ValueError(f"LLM returned invalid JSON: {str(e)}")

    try:
        validated_output = schema(**response_dict)
        logger.info(f"Successfully generated and validated {schema.__name__} (fallback)")
        return validated_output
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        corrected_output = attempt_correction(response_dict, schema, e)
        if corrected_output:
            return corrected_output
        raise


def attempt_correction(data: Dict[str, Any], schema: Type[T], error: ValidationError) -> Optional[T]:
    """
    Attempt to correct validation errors with simple fixes.

    Args:
        data: The data that failed validation
        schema: Target schema
        error: Validation error

    Returns:
        Corrected model instance or None
    """
    try:
        corrected_data = data.copy()

        for err in error.errors():
            field_path = err['loc']
            error_type = err['type']

            if error_type == 'missing':
                field_name = field_path[0] if field_path else None
                if field_name:
                    field_info = schema.model_fields.get(field_name)
                    if field_info:
                        if field_info.default is not None:
                            corrected_data[field_name] = field_info.default
                        elif field_info.annotation == str:
                            corrected_data[field_name] = ""
                        elif field_info.annotation == list:
                            corrected_data[field_name] = []
                        elif field_info.annotation == dict:
                            corrected_data[field_name] = {}

        return schema(**corrected_data)

    except Exception as e:
        logger.warning(f"Correction attempt failed: {e}")
        return None


async def llm_generate_async(
    prompt: str,
    schema: Type[T],
    temperature: Optional[float] = None,
    context: Optional[str] = None,
    client: Optional[LLMClient] = None,
) -> T:
    """
    Async wrapper for llm_generate.

    Uses LangChain's async capabilities via ainvoke.
    """
    # Get client
    if client is None:
        from config.settings import get_settings
        settings = get_settings()
        client = get_llm_client(settings)

    return llm_generate(prompt, schema, temperature, context, client)
