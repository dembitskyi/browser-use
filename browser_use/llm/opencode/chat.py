import json
from dataclasses import dataclass
from typing import Any, TypeVar, overload

import httpx
from opencode_ai import AsyncOpencode
from opencode_ai.types.session_chat_params import Part as OpencodePart
from pydantic import BaseModel

from browser_use.llm.base import BaseChatModel
from browser_use.llm.exceptions import ModelProviderError
from browser_use.llm.messages import AssistantMessage, BaseMessage, ContentPartTextParam, SystemMessage, UserMessage
from browser_use.llm.schema import SchemaOptimizer
from browser_use.llm.views import ChatInvokeCompletion, ChatInvokeUsage

T = TypeVar('T', bound=BaseModel)


@dataclass
class ChatOpencode(BaseChatModel):
	"""
	A wrapper around Opencode's API.
	"""

	model: str
	provider_id: str

	# Optional parameters
	base_url: str | None = None
	api_key: str | None = None
	timeout: float | httpx.Timeout | None = None
	max_retries: int = 5
	client_params: dict[str, Any] | None = None

	@property
	def provider(self) -> str:
		return 'opencode'

	@property
	def name(self) -> str:
		return self.model

	def get_client(self) -> AsyncOpencode:
		client_params = self.client_params or {}
		if self.base_url:
			client_params['base_url'] = self.base_url
		if self.api_key:
			client_params['api_key'] = self.api_key
		if self.timeout is not None:
			client_params['timeout'] = self.timeout
		client_params['max_retries'] = self.max_retries
		return AsyncOpencode(**client_params)

	def _convert_messages(self, messages: list[BaseMessage]) -> tuple[str | None, list[OpencodePart]]:
		"""Converts Langchain messages into Opencode system prompt and text parts."""
		system_prompt = []
		text_content = []

		for msg in messages:
			if isinstance(msg, SystemMessage):
				system_prompt.append(msg.text)
			elif isinstance(msg, UserMessage):
				text_content.append(f'User: {msg.text}')
			elif isinstance(msg, AssistantMessage):
				text_content.append(f'Assistant: {msg.text}')

		parts: list[OpencodePart] = []
		if text_content:
			parts.append({'type': 'text', 'text': '\n\n'.join(text_content)})

		return ('\n\n'.join(system_prompt) if system_prompt else None), parts

	@overload
	async def ainvoke(
		self, messages: list[BaseMessage], output_format: None = None, **kwargs: Any
	) -> ChatInvokeCompletion[str]: ...

	@overload
	async def ainvoke(self, messages: list[BaseMessage], output_format: type[T], **kwargs: Any) -> ChatInvokeCompletion[T]: ...

	async def ainvoke(
		self, messages: list[BaseMessage], output_format: type[T] | None = None, **kwargs: Any
	) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
		client = self.get_client()

		try:
			modified_messages = messages.copy()

			# If output format is required, add instructions
			if output_format is not None:
				schema = SchemaOptimizer.create_optimized_json_schema(output_format)
				json_instruction = f'\n\nIMPORTANT: You must respond with ONLY a valid JSON object (no markdown, no code blocks, no explanations) that exactly matches this schema:\n{json.dumps(schema, indent=2)}'

				instruction_added = False
				if modified_messages and isinstance(modified_messages[-1], UserMessage):
					if isinstance(modified_messages[-1].content, str):
						modified_messages[-1].content += json_instruction
						instruction_added = True
					elif isinstance(modified_messages[-1].content, list):
						modified_messages[-1].content.append(ContentPartTextParam(text=json_instruction))
						instruction_added = True

				if not instruction_added:
					modified_messages.append(UserMessage(content=json_instruction))

			system, parts = self._convert_messages(modified_messages)

			if not parts:
				parts = [{'type': 'text', 'text': ' '}]

			# Create a new session for stateless execution
			session = await client.session.create(extra_body={})
			session_id = session.id

			chat_kwargs = {'id': session_id, 'model_id': self.model, 'provider_id': self.provider_id, 'parts': parts}
			if system:
				chat_kwargs['system'] = system

			# Send chat message
			assistant_msg = await client.session.chat(**chat_kwargs)

			# Extract text directly from the assistant message parts
			assistant_text = ''
			if hasattr(assistant_msg, 'parts'):
				for part in getattr(assistant_msg, 'parts', []):
					if isinstance(part, dict):
						if part.get('type') == 'text':
							assistant_text += part.get('text', '')
					else:
						if getattr(part, 'type', '') == 'text':
							assistant_text += getattr(part, 'text', '')

			if not assistant_text and hasattr(assistant_msg, 'text'):
				assistant_text = getattr(assistant_msg, 'text', '')

			# Extract usage stats
			prompt_tokens = 0
			completion_tokens = 0
			if hasattr(assistant_msg, 'tokens') and assistant_msg.tokens:
				prompt_tokens = int(getattr(assistant_msg.tokens, 'input', 0))
				completion_tokens = int(getattr(assistant_msg.tokens, 'output', 0))
			elif hasattr(assistant_msg, 'info') and isinstance(assistant_msg.info, dict) and 'tokens' in assistant_msg.info:
				prompt_tokens = int(assistant_msg.info['tokens'].get('input', 0))
				completion_tokens = int(assistant_msg.info['tokens'].get('output', 0))

			usage = ChatInvokeUsage(
				prompt_tokens=prompt_tokens,
				completion_tokens=completion_tokens,
				total_tokens=prompt_tokens + completion_tokens,
				prompt_cached_tokens=0,
				prompt_cache_creation_tokens=0,
				prompt_image_tokens=0,
			)

			if output_format is None:
				return ChatInvokeCompletion(completion=assistant_text, usage=usage, stop_reason=None)
			else:
				try:
					text = assistant_text.strip()
					if text.startswith('```json') and text.endswith('```'):
						text = text[7:-3].strip()
					elif text.startswith('```') and text.endswith('```'):
						text = text[3:-3].strip()

					parsed_data = json.loads(text)
					parsed = output_format.model_validate(parsed_data)
					return ChatInvokeCompletion(completion=parsed, usage=usage, stop_reason=None)
				except Exception as e:
					raise ModelProviderError(
						message=f'Failed to parse JSON response: {str(e)}. Raw response: {assistant_text[:200]}',
						status_code=500,
						model=self.name,
					) from e

		except Exception as e:
			if isinstance(e, ModelProviderError):
				raise
			raise ModelProviderError(message=str(e), model=self.name) from e
