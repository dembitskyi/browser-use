import json
from dataclasses import dataclass
from typing import Any, TypeVar, overload

import httpx
from opencode_ai import AsyncOpencode
from opencode_ai.types.session_chat_params import Part as OpencodePart
from pydantic import BaseModel

from browser_use.llm.base import BaseChatModel
from browser_use.llm.exceptions import ModelProviderError
from browser_use.llm.messages import (
	BaseMessage,
	ContentPartTextParam,
	SystemMessage,
	UserMessage,
)
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

	def _file_part(self, url: str, media_type: str | None, filename: str | None = None) -> OpencodePart | None:
		"""Map a data/base64 asset to an Opencode `file` part (opencode takes images as file parts)."""
		if not url:
			return None
		# URI scheme (RFC 3986) and MIME type (RFC 2045) are case-insensitive.
		scheme, sep, remainder = url.partition(':')
		if not sep or scheme.lower() not in ('data', 'http', 'https'):
			# Raw base64 with no scheme — wrap using the declared media type.
			url = f'data:{media_type or "application/octet-stream"};base64,{url}'
			scheme, _, remainder = url.partition(':')
		if scheme.lower() == 'data':
			mime = remainder.split(';', 1)[0].split(',', 1)[0].lower() or (media_type or 'application/octet-stream').lower()
		else:
			mime = (media_type or 'application/octet-stream').lower()
		part: OpencodePart = {'type': 'file', 'mime': mime, 'url': url}
		if filename:
			part['filename'] = filename
		return part

	def _tool_calls_text(self, tool_calls: list) -> str:
		return '\n'.join(f'- called {tc.function.name}({tc.function.arguments})' for tc in tool_calls)

	def _part_text(self, part: Any) -> str:
		"""Text for a text/refusal content part; '' for media (emitted as file parts)."""
		ptype = getattr(part, 'type', None)
		if ptype == 'text':
			return part.text
		if ptype == 'refusal':
			return f'[Refusal] {part.refusal}'
		return ''

	def _convert_messages(self, messages: list[BaseMessage]) -> tuple[str | None, list[OpencodePart]]:
		"""Convert browser-use messages into an Opencode system prompt plus text/file parts."""
		system_prompt: list[str] = []
		text_content: list[str] = []
		file_parts: list[OpencodePart] = []

		for msg in messages:
			role = getattr(msg, 'role', 'user')
			if isinstance(msg, SystemMessage):
				body = msg.text or ''
				if body:
					system_prompt.append(body)
				continue

			chunks: list[str] = []
			content = getattr(msg, 'content', None)
			if isinstance(content, str):
				if content:
					chunks.append(content)
			elif isinstance(content, list):
				for part in content:
					ptype = getattr(part, 'type', None)
					if ptype == 'image_url':
						fp = self._file_part(part.image_url.url, getattr(part.image_url, 'media_type', None))
						if fp:
							file_parts.append(fp)
					elif ptype in ('file', 'input_audio'):
						url = getattr(part, 'url', None) or getattr(part, 'data', None)
						fp = self._file_part(url, getattr(part, 'media_type', None), getattr(part, 'filename', None))
						if fp:
							file_parts.append(fp)
					else:
						t = self._part_text(part)
						if t:
							chunks.append(t)

			if role == 'assistant':
				for attr in ('tool_calls',):
					for tc in getattr(msg, attr, None) or []:
						fn = getattr(tc, 'function', tc)
						chunks.append(f'[Assistant tool call] {getattr(fn, "name", "?")}({getattr(fn, "arguments", "")})')
				if getattr(msg, 'refusal', None):
					chunks.append(f'[Refusal] {msg.refusal}')

			body = '\n'.join(c for c in chunks if c)
			if body:
				text_content.append(f'{role.capitalize()}: {body}')

		parts: list[OpencodePart] = []
		if text_content:
			parts.append({'type': 'text', 'text': '\n\n'.join(text_content)})
		parts.extend(file_parts)

		return ('\n\n'.join(system_prompt) if system_prompt else None), parts

	def _extract_usage(self, assistant_msg: Any) -> ChatInvokeUsage:
		"""Map Opencode Tokens (input/output/reasoning + cache read/write) to ChatInvokeUsage."""
		tokens = getattr(assistant_msg, 'tokens', None)
		prompt = completion = cached = cache_write = reasoning = 0
		if tokens:
			prompt = int(getattr(tokens, 'input', 0) or 0)
			completion = int(getattr(tokens, 'output', 0) or 0)
			reasoning = int(getattr(tokens, 'reasoning', 0) or 0)
			cache = getattr(tokens, 'cache', None)
			if cache:
				cached = int(getattr(cache, 'read', 0) or 0)
				cache_write = int(getattr(cache, 'write', 0) or 0)
		completion += reasoning
		return ChatInvokeUsage(
			prompt_tokens=prompt,
			completion_tokens=completion,
			total_tokens=prompt + completion,
			prompt_cached_tokens=cached,
			prompt_cache_creation_tokens=cache_write,
			prompt_image_tokens=0,
		)

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
			# Never mutate the caller's messages: rebuild the last user message.
			modified_messages = list(messages)

			# If output format is required, add instructions
			if output_format is not None:
				schema = SchemaOptimizer.create_optimized_json_schema(output_format)
				json_instruction = f'\n\nIMPORTANT: You must respond with ONLY a valid JSON object (no markdown, no code blocks, no explanations) that exactly matches this schema:\n{json.dumps(schema, indent=2)}'

				if modified_messages and isinstance(modified_messages[-1], UserMessage):
					last = modified_messages[-1]
					if isinstance(last.content, str):
						new_content: Any = last.content + json_instruction
					elif isinstance(last.content, list):
						new_content = [*last.content, ContentPartTextParam(text=json_instruction)]
					else:
						new_content = last.content
					modified_messages[-1] = last.model_copy(update={'content': new_content})
				else:
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

			# Surface provider/abort errors instead of failing cryptically on empty output.
			err = getattr(assistant_msg, 'error', None)
			if err:
				msg = getattr(err, 'message', None) or getattr(err, 'name', None) or (err if isinstance(err, str) else str(err))
				raise ModelProviderError(message=f'Opencode returned an error: {msg}', status_code=500, model=self.name)

			# Extract text from the assistant message parts
			assistant_text = ''
			for part in getattr(assistant_msg, 'parts', []) or []:
				if isinstance(part, dict):
					if part.get('type') == 'text':
						assistant_text += part.get('text', '')
				elif getattr(part, 'type', '') == 'text':
					assistant_text += getattr(part, 'text', '')

			usage = self._extract_usage(assistant_msg)

			if output_format is None:
				return ChatInvokeCompletion(completion=assistant_text, usage=usage, stop_reason='stop')

			try:
				parsed = output_format.model_validate_json(assistant_text)
				return ChatInvokeCompletion(completion=parsed, usage=usage, stop_reason='stop')
			except Exception as e:
				raise ModelProviderError(
					message=f'Failed to parse structured response: {e}. Raw response: {assistant_text[:300]}',
					status_code=500,
					model=self.name,
				) from e

		except Exception as e:
			if isinstance(e, ModelProviderError):
				raise
			raise ModelProviderError(message=str(e), model=self.name) from e
