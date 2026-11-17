"""本地模型版本仓库：完整产物写完后发布，运行状态只保存可搬移的引用。"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable
import json
import os
import re

from harness.core.contracts import digest
from harness.adapters.storage.local import LocalStore, atomic_json


class ModelArtifacts:
    """持有本次运行自己生成的模型文件；调用者负责维护目录的可信来源。

    哈希检查文件是否与已确认版本一致，不验证文件作者身份。后端使用
    joblib时只能加载本项目可信运行的产物。一个仓库通过本地锁顺序写入；
    发布前退出留下的.writing目录不会成为可加载版本。
    """
    format = "pipeline_bundle_v1"

    def __init__(self, folder: Path):
        self.folder = Path(folder).resolve()
        self.folder.mkdir(parents=True, exist_ok=True)
        self._lock = LocalStore(self.folder)

    def _directory(self, key: str) -> Path:
        if not isinstance(key, str) or not re.fullmatch(r"[0-9a-f]{64}", key):
            raise ValueError("模型版本键必须为SHA256")
        path = self.folder / key
        if path.resolve().parent != self.folder:
            raise ValueError("模型版本超出当前仓库")
        return path

    def _read(self, path: Path) -> bytes:
        if not path.resolve().is_relative_to(self.folder):
            raise ValueError("模型文件超出当前仓库")
        return path.read_bytes()

    def _reference(self, key: str) -> dict:
        raw = self._read(self._directory(key) / "manifest.json")
        return {"format": self.format, "key": key, "manifest_sha256": sha256(raw).hexdigest()}

    def read(self, reference: dict) -> tuple[dict, bytes]:
        """核对已确认清单和完整内容后返回字节，不在本层反序列化模型。"""
        if set(reference) != {"format", "key", "manifest_sha256"} or reference["format"] != self.format:
            raise ValueError("未知模型产物引用")
        folder = self._directory(reference["key"])
        raw = self._read(folder / "manifest.json")
        if sha256(raw).hexdigest() != reference["manifest_sha256"]:
            raise ValueError("模型清单与已确认版本不一致")
        manifest = json.loads(raw)
        if manifest.get("format") != self.format or digest(manifest["request"]) != reference["key"]:
            raise ValueError("模型清单与训练请求不一致")
        content = self._read(folder / "pipeline.bin")
        if len(content) != manifest["size"] or sha256(content).hexdigest() != manifest["sha256"]:
            raise ValueError("模型内容不完整或已变化")
        return manifest, content

    def find(self, request: dict) -> dict | None:
        """只查询并核对完整发布的同请求产物；不构造或重新拟合模型。"""
        key = digest(request)
        if not self._directory(key).exists():
            return None
        reference = self._reference(key)
        manifest, _ = self.read(reference)
        if digest(manifest["request"]) != digest(request):
            raise ValueError("已发布产物与待核对请求不同")
        return reference

    def create(self, request: dict, produce: Callable[[], bytes]) -> dict:
        """同一请求复用已发布版本；produce失败时保持已有版本不变。

        锁覆盖查询、生成和发布，防止两个遵循本协议的写入者重复执行。
        同一文件系统内先写产物和清单，再重命名整个目录；返回后控制器
        才能将引用登记到运行状态。运行状态中未确认的任务仍由控制器处理。
        """
        # 固定本次请求，避免生产回调修改来源后发布到错误的版本键。
        request = json.loads(json.dumps(request, ensure_ascii=False, allow_nan=False))
        key = digest(request)
        destination = self._directory(key)
        with self._lock.exclusive():
            if destination.exists():
                reference = self._reference(key)
                self.read(reference)
                return reference
            content = produce()
            if not isinstance(content, bytes) or not content:
                raise ValueError("模型序列化必须返回非空字节")
            with TemporaryDirectory(prefix=".writing-", dir=self.folder) as scratch:
                stage = Path(scratch) / "version"
                stage.mkdir()
                with (stage / "pipeline.bin").open("xb") as handle:
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
                atomic_json(stage / "manifest.json", {
                    "format": self.format, "request": request,
                    "size": len(content), "sha256": sha256(content).hexdigest(),
                })
                stage.rename(destination)
            return self._reference(key)
