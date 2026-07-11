# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# 《从零构建推理模型》配套源码：https://mng.bz/lZ5B
# 代码仓库：https://github.com/rasbt/reasoning-from-scratch

from pathlib import Path
import sys
import requests
from urllib.parse import urlparse


def _download_error_message(filename, url, primary_error, backup_url=None, backup_error=None):
    details = [f"下载 {filename} 失败。"]

    if primary_error is not None:
        details.append(
            f"主 URL 请求失败（{url}）："
            f"{type(primary_error).__name__}: {primary_error}"
        )

    if backup_url and backup_error is not None:
        details.append(
            f"备用 URL 请求失败（{backup_url}）："
            f"{type(backup_error).__name__}: {backup_error}"
        )

    cert_or_proxy_issue = any(
        isinstance(err, (requests.exceptions.ProxyError, requests.exceptions.SSLError))
        for err in (primary_error, backup_error)
        if err is not None
    )
    if not cert_or_proxy_issue:
        lowered = " ".join(
            str(err).lower() for err in (primary_error, backup_error) if err is not None
        )
        cert_or_proxy_issue = any(
            keyword in lowered for keyword in ("certificate", "ssl", "tls", "proxy")
        )

    if cert_or_proxy_issue:
        details.append(
            "这可能发生在单位或学校的计算机上：VPN、代理或防病毒工具"
            "拦截了 HTTPS 证书。"
        )

    details.append(
        "请参阅故障排除指南："
        "https://github.com/rasbt/reasoning-from-scratch/blob/main/troubleshooting.md "
        "（尤其是 'File Download Issues' 一节）。"
    )
    return "\n".join(details)


def download_file(url, out_dir=".", backup_url=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(urlparse(url).path).name
    dest = out_dir / filename

    def try_download(u):
        try:
            with requests.get(u, stream=True, timeout=30) as r:
                r.raise_for_status()
                size_remote = int(r.headers.get("Content-Length", 0))

                # 如果下载已完成，则跳过
                if dest.exists() and size_remote and dest.stat().st_size == size_remote:
                    print(f"✓ {dest} 已是最新版本")
                    return True, None

                # 以 1 MiB 为分块下载并显示进度
                block = 1024 * 1024
                downloaded = 0
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=block):
                        if not chunk:
                            continue
                        f.write(chunk)
                        downloaded += len(chunk)
                        if size_remote:
                            pct = downloaded * 100 // size_remote
                            sys.stdout.write(
                                f"\r{filename}: {pct:3d}% "
                                f"({downloaded // (1024*1024)} MiB / "
                                f"{size_remote // (1024*1024)} MiB)"
                            )
                            sys.stdout.flush()
                if size_remote:
                    sys.stdout.write("\n")
            return True, None
        except requests.RequestException as exc:
            return False, exc

    # 先尝试主 URL
    success, primary_error = try_download(url)
    if success:
        return dest

    # 如果提供了备用 URL，则尝试备用地址
    backup_error = None
    if backup_url:
        print(f"主 URL（{url}）请求失败。\n正在尝试备用 URL（{backup_url}）……")
        success, backup_error = try_download(backup_url)
        if success:
            return dest

    message = _download_error_message(
        filename=filename,
        url=url,
        primary_error=primary_error,
        backup_url=backup_url,
        backup_error=backup_error,
    )
    raise RuntimeError(message) from (backup_error or primary_error)
