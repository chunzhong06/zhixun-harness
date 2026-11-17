"""PyTRiO的HTTP传输边界；模拟器与真实连接共用请求编码及回复检查。"""
from __future__ import annotations

import json
import os
import re
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from harness.core.contracts import digest
from harness.adapters.providers.pytrio_planner import ChatRequest, ChatResponse, PyTrioConfig


class TransportError(RuntimeError):
    """只携带本地错误类别；远端错误仅允许经过脱敏的结构化诊断。"""


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # 认证请求固定发送给配置中的地址，重定向由配置维护者处理。
        return None


class HttpChatTransport:
    """以一次POST发送一个计划请求，不自动重复可能已执行的推理。

    api_key由调用方显式提供，本层不读取环境或账户文件。wire只用于注入
    标明simulated=True的模拟HTTP边界；省略时需显式allow_network才可发送。
    timeout为底层连接/读取超时，非整个推理任务的强制终止时间。
    """
    MAX_RESPONSE_BYTES = 1024 * 1024

    def __init__(self, config: PyTrioConfig, api_key: str, *, wire=None, allow_network=False):
        if not isinstance(api_key, str) or not api_key.strip() or any(c in api_key for c in "\r\n"):
            raise ValueError("认证凭据缺失或格式无效")
        if wire is not None and getattr(wire, "simulated", None) is not True:
            raise ValueError("注入传输必须明确标记为模拟服务")
        self.endpoint = config.base_url.rstrip("/") + ("/completions" if config.raw_completion else "/chat/completions")
        self._api_key, self._wire = api_key, wire
        self.network_enabled = wire is None
        self.allow_network = allow_network
        self.identity = "pytrio-http-v1:" + digest([self.endpoint, wire.identity if wire else "urllib"])

    def send(self, request: ChatRequest) -> ChatResponse:
        budget = os.environ.get('HARNESS_SERVICE_BUDGET')
        if budget and self.network_enabled:
            from harness.engine.service_budget import ServiceBudget
            return ServiceBudget(budget).send(request, self._send)
        return self._send(request)

    def _send(self, request: ChatRequest) -> ChatResponse:
        if request.url != self.endpoint:
            raise TransportError("endpoint_mismatch")
        if self.network_enabled and not self.allow_network:
            raise TransportError("network_disabled")
        body = json.dumps(request.body, ensure_ascii=False, allow_nan=False).encode("utf-8")
        http = Request(self.endpoint, data=body, method="POST", headers={
            "Authorization": "Bearer " + self._api_key, "Content-Type": "application/json",
            "Accept": "application/json"})
        opener = self._wire if self._wire is not None else build_opener(_NoRedirect())
        try:
            with opener.open(http, timeout=request.timeout_seconds) as response:
                status = response.status
                if status != 200:
                    return ChatResponse(status, self._safe_error(response.read(8192)))
                raw = response.read(self.MAX_RESPONSE_BYTES + 1)
        except HTTPError as exc:
            status = exc.code
            try:
                detail = self._safe_error(exc.read(8192))
            finally:
                exc.close()
            return ChatResponse(status, detail)
        except TimeoutError:
            raise TransportError("timeout") from None
        except Exception:
            raise TransportError("connection_failed") from None
        if len(raw) > self.MAX_RESPONSE_BYTES:
            raise TransportError("response_too_large")
        try:
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError()
        except (ValueError, UnicodeError):
            raise TransportError("invalid_response_json") from None
        return ChatResponse(status, payload)

    def _safe_error(self, raw):
        """只取JSON错误的标量字段；不保存HTML、请求头或任意服务调试对象。"""
        try:
            payload = json.loads(raw.decode('utf-8'))
        except (ValueError, UnicodeError):
            return {}
        if not isinstance(payload, dict):
            return {}
        error = payload.get('error', payload)
        if isinstance(error, str):
            error = {'message': error}
        if not isinstance(error, dict):
            return {}
        result = {}
        for field in ('type', 'code', 'message'):
            value = error.get(field)
            if isinstance(value, (str, int)):
                value = str(value).replace(self._api_key, '[已隐藏凭据]')
                value = re.sub(r'(?i)Bearer\s+[^\s\"\',;]+', 'Bearer [已隐藏凭据]', value)
                value = re.sub(r'(?i)(?:sk-|trio[-_])[A-Za-z0-9_-]{8,}', '[已隐藏凭据]', value)
                result[field] = value[:512]
        return {'error': result} if result else {}
