"""
Google Drive sync service for ApexQuant.

Downloads feature CSVs and model files from shared Google Drive links
using a custom ``requests``-based downloader (no ``gdown`` dependency),
caching them locally for the data loader and backtest engine.

Usage::

    from services.drive_sync import DriveSync

    ds = DriveSync()
    result = ds.sync_data_folder("https://drive.google.com/drive/folders/1abc...")
    print(result)  # {"status": "ok", "files_downloaded": 3, "files": [...]}

    ds.sync_model_file(
        "https://drive.google.com/file/d/1xyz.../view",
        model_name="vol_lgb.txt",
    )
"""

__all__ = ["DriveSync"]

import json as _json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import requests
from loguru import logger


# ---------------------------------------------------------------------------
# URL parsing helpers
# ---------------------------------------------------------------------------

_FOLDER_RE = re.compile(r"/folders/([a-zA-Z0-9_-]+)")
_FILE_D_RE = re.compile(r"/file/d/([a-zA-Z0-9_-]+)")
_OPEN_ID_RE = re.compile(r"[?&]id=([a-zA-Z0-9_-]+)")

# Used to extract the real filename from Content-Disposition header
_CD_FILENAME_RE = re.compile(r'filename\*?=["\']?(?:UTF-8\'\')?([^"\';\r\n]+)')

# ---------------------------------------------------------------------------
# Low-level Google Drive download helpers (no gdown)
# ---------------------------------------------------------------------------

def _extract_filename_from_cd(headers: dict) -> str | None:
    """Parse filename from a Content-Disposition header."""
    cd = headers.get("Content-Disposition", "")
    if not cd:
        return None
    m = _CD_FILENAME_RE.search(cd)
    return m.group(1) if m else None


def _gdrive_download_file(file_id: str, dest: Path) -> Path:
    """Download a single Google Drive file by ID using requests.

    Handles the virus-scan confirmation page that Google shows for
    large files.  Streams the response to *dest*.

    Args:
        file_id: The Google Drive file ID.
        dest: Target file path (parent dirs are created automatically).

    Returns:
        The *dest* path on success.

    Raises:
        RuntimeError: On download failure, permission errors, or
            suspiciously small output.
    """
    session = requests.Session()
    url = f"https://drive.google.com/uc?export=download&id={file_id}"

    logger.info("Downloading Drive file id={} ...", file_id)
    resp = session.get(url, stream=True, timeout=120)
    resp.raise_for_status()

    # Google may return an HTML virus-scan confirmation page
    content_type = resp.headers.get("Content-Type", "")
    if "text/html" in content_type:
        body = resp.text
        # Look for confirm token
        token_match = re.search(
            r'confirm=([0-9A-Za-z_-]+)', body,
        )
        if token_match:
            confirm_url = (
                f"https://drive.google.com/uc?export=download"
                f"&confirm={token_match.group(1)}&id={file_id}"
            )
            resp = session.get(confirm_url, stream=True, timeout=120)
            resp.raise_for_status()
        else:
            # Also try the uuid-based confirmation used on newer pages
            uuid_match = re.search(r'name="uuid" value="([^"]+)"', body)
            if uuid_match:
                confirm_url = (
                    f"https://drive.google.com/uc?export=download"
                    f"&id={file_id}&uuid={uuid_match.group(1)}"
                )
                resp = session.get(confirm_url, stream=True, timeout=120)
                resp.raise_for_status()
            else:
                # Check if this is truly an error page
                if "ServiceLogin" in body or "accounts.google.com" in body:
                    raise RuntimeError(
                        "Google Drive requires sign-in — "
                        "make the file publicly accessible"
                    )
                raise RuntimeError(
                    "Unrecognised Google Drive confirmation page — "
                    "make the file publicly accessible and try again"
                )

    # Try to get real filename from Content-Disposition
    real_name = _extract_filename_from_cd(resp.headers)

    # Stream to disk
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=32_768):
            f.write(chunk)

    size = dest.stat().st_size
    logger.info("Downloaded {} ({} bytes)", dest.name, size)

    # Sanity check — very small files are usually error pages
    if size < 200:
        with open(dest, "rb") as fh:
            head = fh.read(200)
        if head.lstrip().lower().startswith((b"<!doctype", b"<html")):
            dest.unlink(missing_ok=True)
            raise RuntimeError(
                f"Downloaded file is an HTML error page ({size} bytes) "
                "— likely a permission error"
            )

    # If Content-Disposition gave a real filename, rename
    if real_name and dest.name != real_name:
        renamed = dest.parent / real_name
        shutil.move(str(dest), str(renamed))
        logger.info("Renamed {} -> {}", dest.name, real_name)
        return renamed

    return dest


def _gdrive_list_folder(folder_id: str) -> list[dict]:
    """List files in a public Google Drive folder using Drive API v3.

    Args:
        folder_id: The Google Drive folder ID.

    Returns:
        List of dicts with ``id``, ``name``, and ``mimeType`` keys.
    """
    from config.default import CONFIG

    api_key = CONFIG.get("GDRIVE_API_KEY", "")
    if not api_key:
        raise RuntimeError("GDRIVE_API_KEY not set in config/default.py")

    files: list[dict] = []
    page_token = None
    while True:
        params: dict[str, Any] = {
            "q": f"'{folder_id}' in parents and trashed=false",
            "fields": "nextPageToken, files(id, name, mimeType)",
            "key": api_key,
            "pageSize": 100,
        }
        if page_token:
            params["pageToken"] = page_token
        resp = requests.get(
            "https://www.googleapis.com/drive/v3/files",
            params=params,
            timeout=30,
        )
        if resp.status_code == 403:
            raise RuntimeError(
                "Drive API 403 — check API key is valid and "
                "Google Drive API is enabled"
            )
        resp.raise_for_status()
        data = resp.json()
        files.extend(data.get("files", []))
        page_token = data.get("nextPageToken")
        if not page_token:
            break

    logger.info("Drive API listed {} files in folder {}", len(files), folder_id)
    return files


_GDRIVE_FOLDER_MIME = "application/vnd.google-apps.folder"
_GDRIVE_NATIVE_PREFIX = "application/vnd.google-apps."


def _gdrive_download_folder(
    folder_id: str, dest_dir: Path
) -> list[Path]:
    """Recursively download all files from a public Google Drive folder.

    Mirrors the remote folder structure locally.  Sub-folders are
    created and recursed into.  Google-native formats (Sheets, Docs,
    Slides, etc.) are skipped since they cannot be downloaded as
    binary files.

    Args:
        folder_id: Google Drive folder ID.
        dest_dir: Local directory to save files into.

    Returns:
        List of all downloaded file paths (across all nesting levels).
    """
    entries = _gdrive_list_folder(folder_id)
    if not entries:
        logger.warning("No files found in folder {}", folder_id)
        return []

    dest_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []

    for entry in entries:
        fid = entry["id"]
        fname = entry.get("name", fid)
        mime = entry.get("mimeType", "")

        if mime == _GDRIVE_FOLDER_MIME:
            # Recurse into sub-folder
            sub_dir = dest_dir / fname
            logger.info("Entering sub-folder: {} -> {}", fname, sub_dir)
            sub_files = _gdrive_download_folder(fid, sub_dir)
            downloaded.extend(sub_files)
        elif mime.startswith(_GDRIVE_NATIVE_PREFIX):
            # Skip Google-native formats (Sheets, Docs, Slides, etc.)
            logger.debug(
                "Skipping Google-native file {} ({})", fname, mime,
            )
        else:
            target = dest_dir / fname
            try:
                result = _gdrive_download_file(fid, target)
                downloaded.append(result)
            except Exception as exc:
                logger.warning(
                    "Failed to download file {} ({}): {}",
                    fid, fname, exc,
                )

    return downloaded


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class DriveSync:
    """Download data and model files from Google Drive.

    Uses a custom ``requests``-based downloader for public / link-shared
    Google Drive folders and files.  Downloaded files are cached in the
    configured local directories.

    Attributes:
        data_dir: Local directory for feature CSVs.
        models_dir: Local directory for model weight files.
    """

    _MODEL_EXTENSIONS = {".pt", ".pth", ".txt", ".joblib", ".pkl", ".bin"}

    def __init__(
        self,
        data_dir: str = "data/features",
        models_dir: str = "models",
    ) -> None:
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sync_data_folder(self, folder_url: str) -> dict[str, Any]:
        """Download all CSV files from a Drive folder.

        Args:
            folder_url: Google Drive folder URL.

        Returns:
            Dict with status, count, and file list.
        """
        logger.info("Syncing data folder: {}", folder_url)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        try:
            folder_id = self._extract_drive_id(folder_url)
            downloaded = _gdrive_download_folder(folder_id, self.data_dir)
            files = [str(p) for p in downloaded]
            logger.info("Downloaded {} files to {}", len(files), self.data_dir)

            return {
                "status": "ok",
                "files_downloaded": len(files),
                "files": files,
            }

        except Exception as exc:
            logger.error("Drive data sync failed: {}", exc)
            return {
                "status": "error",
                "message": str(exc),
                "files_downloaded": 0,
                "files": [],
            }

    def sync_model_file(
        self, file_url: str, model_name: str
    ) -> dict[str, Any]:
        """Download a single model file from Drive.

        Args:
            file_url: Google Drive file URL.
            model_name: Target filename inside ``models/``.

        Returns:
            Dict with status and local path.
        """
        logger.info("Syncing model file: {} -> {}", file_url, model_name)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.models_dir / model_name

        try:
            file_id = self._extract_drive_id(file_url)
            _gdrive_download_file(file_id, output_path)

            logger.info("Model saved to {}", output_path)
            return {
                "status": "ok",
                "path": str(output_path),
            }

        except Exception as exc:
            logger.error("Model file sync failed: {}", exc)
            return {
                "status": "error",
                "message": str(exc),
                "path": None,
            }

    def sync_model_folder(self, folder_url: str) -> dict[str, Any]:
        """Download all model files from a Drive folder.

        Only files with recognised model extensions (.pt, .txt, .joblib,
        .pkl) are kept.

        Args:
            folder_url: Google Drive folder URL.

        Returns:
            Dict with status, count, and file list.
        """
        logger.info("Syncing models folder: {}", folder_url)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        staging = self.models_dir / "_drive_staging"
        staging.mkdir(parents=True, exist_ok=True)

        try:
            folder_id = self._extract_drive_id(folder_url)
            downloaded = _gdrive_download_folder(folder_id, staging)

            # Move model files from staging to models/
            kept: list[str] = []
            for fpath in downloaded:
                if fpath.suffix.lower() in self._MODEL_EXTENSIONS:
                    try:
                        rel = fpath.relative_to(staging)
                    except ValueError:
                        rel = Path(fpath.name)
                    dest = self.models_dir / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(fpath), str(dest))
                    kept.append(str(dest))
                    logger.info("  Kept model: {}", dest)

            # Also preserve meta.json files alongside weights
            for fpath in downloaded:
                if fpath.name == "meta.json":
                    try:
                        rel = fpath.relative_to(staging)
                    except ValueError:
                        rel = Path(fpath.name)
                    dest = self.models_dir / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    if not dest.exists():
                        shutil.move(str(fpath), str(dest))

            shutil.rmtree(staging, ignore_errors=True)

            logger.info("Synced {} model files to {}", len(kept), self.models_dir)
            return {
                "status": "ok",
                "models_downloaded": len(kept),
                "files": kept,
            }

        except Exception as exc:
            shutil.rmtree(staging, ignore_errors=True)
            logger.error("Models folder sync failed: {}", exc)
            return {
                "status": "error",
                "message": str(exc),
                "models_downloaded": 0,
                "files": [],
            }

    # Extension-to-adapter mapping for auto-generated meta.json
    _EXT_ADAPTER_MAP: dict[str, str] = {
        ".joblib": "lightgbm",
        ".pkl": "lightgbm",
        ".txt": "lightgbm",
        ".bin": "lightgbm",
        ".pt": "multiscale_cnn",
        ".pth": "multiscale_cnn",
    }

    # Filename-keyword → (layer_path, output_label) mapping.
    # Order matters: more specific patterns are checked first.
    _LAYER_RULES: list[tuple[tuple[str, ...], str, str]] = [
        # (required_keywords, layer_path, output_label)
        (("meta", "bottom"), "layer3/trade_filter/lgb_bottom_v1", "meta_bottom_prob"),
        (("meta", "top"),    "layer3/trade_filter/lgb_top_v1",    "meta_top_prob"),
        (("vol",),           "layer1/volatility/lgb_vol_v1",      "vol_prob"),
        (("volatility",),    "layer1/volatility/lgb_vol_v1",      "vol_prob"),
        (("bottom",),        "layer2/tp_bottom/cnn_bottom_v1",    "tp_bottom_prob"),
        (("tp_bottom",),     "layer2/tp_bottom/cnn_bottom_v1",    "tp_bottom_prob"),
        (("top",),           "layer2/tp_top/cnn_top_v1",          "tp_top_prob"),
        (("tp_top",),        "layer2/tp_top/cnn_top_v1",          "tp_top_prob"),
        (("meta",),          "layer3/trade_filter/lgb_meta_v1",   "meta_prob"),
    ]

    @staticmethod
    def _is_file_url(url: str) -> bool:
        """Return True if *url* points to a single Drive file."""
        return "/file/d/" in url or "uc?id=" in url or "uc?export=" in url

    @classmethod
    def _infer_layer(cls, filename: str) -> tuple[str, str]:
        """Infer (layer_path, output_label) from a filename.

        Checks filename against keyword rules. Returns a fallback path
        if no rule matches.
        """
        name_lower = filename.lower().replace("-", "_")
        for keywords, layer_path, output_label in cls._LAYER_RULES:
            if all(kw in name_lower for kw in keywords):
                return layer_path, output_label
        stem = Path(filename).stem
        return f"layer1/unknown_{stem}", f"{stem}_prob"

    @staticmethod
    def _detect_file_type(fpath: Path) -> str | None:
        """Detect model file type from magic bytes and trial loading.

        Detection order:
        1. HTML content check → Google Drive permission error page
        2. PK\\x03\\x04 header → ``'pytorch'`` (PyTorch ZIP format)
        3. Pickle header (0x80 0x02/04/05) → try torch then joblib
        4. Brute-force trial load with torch, then joblib

        Returns:
            ``'pytorch'``, ``'joblib'``, or ``None``.
        """
        if not fpath.is_file() or fpath.stat().st_size == 0:
            return None

        with open(fpath, "rb") as fh:
            header = fh.read(512)

        logger.info(
            "_detect_file_type: '{}' size={} header={}",
            fpath.name, fpath.stat().st_size, header[:8].hex(),
        )

        # 0. HTML error page check (Drive permission denied)
        if header[:15].lstrip().lower().startswith((b"<!doctype", b"<html")):
            logger.warning(
                "File '{}' is an HTML page — likely a Google Drive "
                "permission error, not a model file",
                fpath.name,
            )
            return None

        # 1. PyTorch new format — ZIP archive (PK\x03\x04)
        if header[:4] == b"PK\x03\x04":
            logger.info("Detected as pytorch via PK\\x03\\x04 header")
            return "pytorch"

        # 2. Pickle protocol header (0x80 0x0N)
        if len(header) >= 2 and header[0] == 0x80 and header[1] in (
            0x02, 0x04, 0x05,
        ):
            try:
                import torch
                torch.load(str(fpath), map_location="cpu", weights_only=True)
                logger.info("Detected as pytorch via pickle header + torch.load")
                return "pytorch"
            except Exception:
                pass
            try:
                import joblib
                joblib.load(str(fpath))
                logger.info("Detected as joblib via pickle header + joblib.load")
                return "joblib"
            except Exception:
                pass

        # 3. Brute-force trial load
        try:
            import torch
            torch.load(str(fpath), map_location="cpu", weights_only=True)
            logger.info("Detected as pytorch via brute-force torch.load")
            return "pytorch"
        except Exception:
            pass

        try:
            import joblib
            joblib.load(str(fpath))
            logger.info("Detected as joblib via brute-force joblib.load")
            return "joblib"
        except Exception:
            pass

        logger.warning(
            "Cannot detect type for '{}' (header={})",
            fpath.name, header[:8].hex(),
        )
        return None

    # Map from _detect_file_type result to canonical extension
    _TYPE_TO_EXT: dict[str, str] = {
        "pytorch": ".pt",
        "joblib": ".joblib",
    }

    # Map from _detect_file_type result to adapter type
    _TYPE_TO_ADAPTER: dict[str, str] = {
        "pytorch": "multiscale_cnn",
        "joblib": "lightgbm",
    }

    def sync_smart_model(self, drive_url: str) -> dict[str, Any]:
        """Download a model from Drive with full auto-detection.

        1. Downloads via custom requests-based downloader.
        2. Walks download directory to find ALL files.
        3. Detects type via ``_detect_file_type`` → ``'pytorch'``/``'joblib'``.
        4. Infers layer placement from filename keywords.
        5. Renames to ``weights.pt`` or ``weights.joblib``.
        6. Auto-generates ``meta.json``.

        Args:
            drive_url: Any Google Drive file or folder URL.

        Returns:
            Dict with status, models list, and debug_files list.
        """
        drive_url = drive_url.strip()
        self.models_dir.mkdir(parents=True, exist_ok=True)

        staging = self.models_dir / "_smart_staging"
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True, exist_ok=True)

        try:
            drive_id = self._extract_drive_id(drive_url)
            if self._is_file_url(drive_url):
                target = staging / drive_id
                _gdrive_download_file(drive_id, target)
            else:
                _gdrive_download_folder(drive_id, staging)
        except Exception as exc:
            shutil.rmtree(staging, ignore_errors=True)
            return {"status": "error", "message": str(exc)}

        results, debug_files = self._process_downloaded_model_files(staging)

        shutil.rmtree(staging, ignore_errors=True)

        ok_results = [r for r in results if r.get("status") == "ok"]
        if not ok_results:
            file_info = "; ".join(
                f"{d['name']} ({d['size']}B, hdr={d['header']})"
                for d in debug_files
            ) or "no files found"
            return {
                "status": "error",
                "message": f"No recognised weight files. Downloaded: {file_info}",
                "debug_files": debug_files,
            }

        return {
            "status": "ok",
            "models": results,
        }

    def _process_downloaded_model_files(
        self, staging: Path
    ) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
        """Process all downloaded files in *staging* into models/.

        Walks the staging tree, detects each file's type, infers layer
        placement, moves weights, and auto-generates meta.json.

        Returns:
            Tuple of (results list, debug_files list).
        """
        all_files: list[Path] = []
        debug_files: list[dict[str, str]] = []
        for root, _dirs, files in os.walk(staging):
            for fname in files:
                fpath = Path(root) / fname
                size = fpath.stat().st_size
                header_hex = ""
                if size > 0:
                    with open(fpath, "rb") as fh:
                        header_hex = fh.read(8).hex()
                logger.info(
                    "Downloaded: {} size={} header={}",
                    fpath, size, header_hex,
                )
                debug_files.append({
                    "name": fname,
                    "size": str(size),
                    "header": header_hex[:16],
                })
                all_files.append(fpath)

        results: list[dict[str, Any]] = []
        for fpath in all_files:
            if not fpath.is_file() or fpath.stat().st_size == 0:
                continue
            if fpath.name == "meta.json":
                continue

            original_name = fpath.name
            ext = fpath.suffix.lower()

            # If extension already recognised, map to type string
            if ext in self._EXT_ADAPTER_MAP:
                file_type = "pytorch" if ext in (".pt", ".pth") else "joblib"
            else:
                file_type = self._detect_file_type(fpath)

            if file_type is None:
                with open(fpath, "rb") as fh:
                    hdr = fh.read(8).hex()
                logger.warning(
                    "Cannot identify '{}' (size={}, header={})",
                    original_name, fpath.stat().st_size, hdr,
                )
                results.append({
                    "status": "error",
                    "original_filename": original_name,
                    "message": f"Unrecognised format (header: {hdr[:8]})",
                })
                continue

            detected_ext = self._TYPE_TO_EXT[file_type]
            adapter_type = self._TYPE_TO_ADAPTER[file_type]

            # Use the folder structure from Drive (mirrored in staging)
            # when the file is nested; fall back to keyword heuristics
            # only when the file sits directly in the staging root.
            rel_to_staging = fpath.relative_to(staging)
            if len(rel_to_staging.parts) > 1:
                # File is inside sub-folders — mirror Drive hierarchy
                layer_path = str(rel_to_staging.parent)
                # Derive output_label from the deepest folder name
                deepest = rel_to_staging.parent.name
                output_label = f"{deepest.replace('-', '_')}_prob"
            else:
                # File sits directly in staging root — guess from name
                layer_path, output_label = self._infer_layer(original_name)

            dest_dir = self.models_dir / layer_path
            dest_dir.mkdir(parents=True, exist_ok=True)

            final_path = dest_dir / f"weights{detected_ext}"
            shutil.move(str(fpath), str(final_path))
            logger.info("Placed {} -> {}", original_name, final_path)

            # Companion meta.json from download
            companion_meta = fpath.parent / "meta.json"
            meta_path = dest_dir / "meta.json"
            meta_generated = False

            if companion_meta.exists() and not meta_path.exists():
                shutil.move(str(companion_meta), str(meta_path))
            elif not meta_path.exists():
                name = layer_path.replace("/", "_").replace("\\", "_")
                meta = {
                    "adapter": adapter_type,
                    "output_label": output_label,
                    "name": name,
                }
                meta_path.write_text(
                    _json.dumps(meta, indent=2), encoding="utf-8"
                )
                meta_generated = True
                logger.info("Auto-generated {}", meta_path)

            results.append({
                "status": "ok",
                "original_filename": original_name,
                "inferred_path": layer_path,
                "adapter_type": adapter_type,
                "output_label": output_label,
                "weights_path": str(final_path),
                "meta_generated": meta_generated,
            })

        return results, debug_files

    def sync_smart_models(self, urls: list[str]) -> list[dict[str, Any]]:
        """Run :meth:`sync_smart_model` for each URL in *urls*.

        Returns:
            List of per-URL result dicts.
        """
        return [self.sync_smart_model(url) for url in urls]

    def sync_smart_data(self, urls: list[str]) -> list[dict[str, Any]]:
        """Download data from Drive URLs into ``data_dir/``.

        Args:
            urls: List of Google Drive URLs.

        Returns:
            List of per-URL result dicts with status, path, file count.
        """
        results: list[dict[str, Any]] = []
        self.data_dir.mkdir(parents=True, exist_ok=True)

        for url in urls:
            url = url.strip()
            if not url:
                continue
            try:
                drive_id = self._extract_drive_id(url)
                if self._is_file_url(url):
                    target = self.data_dir / drive_id
                    result_path = _gdrive_download_file(drive_id, target)
                    result = self._process_downloaded_data_files(result_path)
                    results.append(result)
                else:
                    downloaded = _gdrive_download_folder(
                        drive_id, self.data_dir
                    )
                    csv_count = sum(
                        1 for p in downloaded
                        if p.suffix.lower() == ".csv"
                    )
                    results.append({
                        "status": "ok",
                        "path": str(self.data_dir),
                        "files": len(downloaded),
                        "csv_files": csv_count,
                    })
            except Exception as exc:
                logger.error("Failed to sync data URL: {}", exc)
                results.append({"status": "error", "message": str(exc)})

        return results

    @staticmethod
    def _process_downloaded_data_files(fpath: Path) -> dict[str, Any]:
        """Post-process a downloaded data file.

        Checks that the file looks like valid CSV data (not an HTML
        error page).

        Args:
            fpath: Path to the downloaded file.

        Returns:
            Result dict with status, path, filename, files count.
        """
        fname = fpath.name

        if fpath.is_file() and fpath.stat().st_size > 0:
            with open(fpath, "rb") as fh:
                head = fh.read(64)
            if head.lstrip().lower().startswith((b"<!doctype", b"<html")):
                logger.warning(
                    "Downloaded '{}' is HTML (Drive permission error?)",
                    fname,
                )
                fpath.unlink(missing_ok=True)
                return {
                    "status": "error",
                    "message": f"'{fname}' is an HTML error page, not data",
                }

        logger.info("Data file ready: {}", fpath)
        return {
            "status": "ok",
            "path": str(fpath),
            "files": 1,
            "filename": fname,
        }

    # ------------------------------------------------------------------
    # Legacy sync methods (use the requests-based downloader)
    # ------------------------------------------------------------------

    def sync_all_models(
        self, models_map: dict[str, str] | None = None
    ) -> dict[str, Any]:
        """Download model files or folders listed in *models_map*.

        Each key is a relative path under ``models/`` (e.g.
        ``"layer1/volatility/lightgbm_v3"``), and the value is a
        Google Drive URL.

        Args:
            models_map: Mapping of model sub-path to Drive URL.

        Returns:
            Dict mapping each sub-path to its sync result.
        """
        if models_map is None:
            from config.default import CONFIG
            models_map = CONFIG.get("drive", {}).get("models_map", {})

        results: dict[str, Any] = {}
        for rel_path, drive_url in models_map.items():
            dest = self.models_dir / rel_path
            logger.info("Syncing model: {} -> {}", rel_path, dest)

            try:
                drive_id = self._extract_drive_id(drive_url)
                if self._is_file_url(drive_url):
                    results[rel_path] = self._sync_single_file(
                        drive_id, dest
                    )
                else:
                    dest.mkdir(parents=True, exist_ok=True)
                    _gdrive_download_folder(drive_id, dest)
                    results[rel_path] = {"status": "ok", "path": str(dest)}
            except Exception as exc:
                logger.error("Failed to sync {}: {}", rel_path, exc)
                results[rel_path] = {"status": "error", "message": str(exc)}

        return results

    def _sync_single_file(
        self, file_id: str, dest_dir: Path
    ) -> dict[str, Any]:
        """Download a single Drive file into *dest_dir* (legacy path)."""
        dest_dir.mkdir(parents=True, exist_ok=True)

        tmp_path = dest_dir / "_download_tmp"
        downloaded = _gdrive_download_file(file_id, tmp_path)

        ext = downloaded.suffix.lower() if downloaded.suffix else ".bin"

        final_name = f"weights{ext}"
        final_path = dest_dir / final_name
        shutil.move(str(downloaded), str(final_path))

        meta_generated = False
        meta_path = dest_dir / "meta.json"
        if not meta_path.exists():
            adapter_type = self._EXT_ADAPTER_MAP.get(ext, "lightgbm")
            rel = dest_dir.relative_to(self.models_dir)
            name = str(rel).replace("/", "_").replace("\\", "_")
            meta = {
                "adapter": adapter_type,
                "output_label": f"{name}_prob",
                "name": name,
            }
            meta_path.write_text(
                _json.dumps(meta, indent=2), encoding="utf-8"
            )
            meta_generated = True

        return {
            "status": "ok",
            "path": str(final_path),
            "meta_generated": meta_generated,
        }

    def sync_all_data(
        self, data_map: dict[str, str] | None = None
    ) -> dict[str, Any]:
        """Download data folders listed in *data_map*.

        Each key is a local subdirectory name (placed under
        ``data_dir/{name}/``), and the value is a Google Drive folder
        URL.  If *data_map* is ``None``, reads from
        ``CONFIG["drive"]["data_map"]``.

        Args:
            data_map: Mapping of local name to Drive URL.

        Returns:
            Dict with per-source sync results.
        """
        if data_map is None:
            from config.default import CONFIG
            data_map = CONFIG.get("drive", {}).get("data_map", {})

        results: dict[str, Any] = {}
        for name, folder_url in data_map.items():
            dest = self.data_dir / name
            dest.mkdir(parents=True, exist_ok=True)
            logger.info("Syncing data source: {} -> {}", name, dest)

            try:
                folder_id = self._extract_drive_id(folder_url)
                downloaded = _gdrive_download_folder(folder_id, dest)
                n = len(downloaded)
                results[name] = {
                    "status": "ok",
                    "path": str(dest),
                    "files": n,
                }
            except Exception as exc:
                logger.error("Failed to sync data '{}': {}", name, exc)
                results[name] = {"status": "error", "message": str(exc)}

        return results

    def get_cache_status(self) -> dict[str, Any]:
        """Summarise locally cached data and model files.

        Returns:
            Dict with ``data_files`` and ``model_files`` lists, each
            containing dicts with ``name``, ``size_kb``, and ``modified``.
        """
        def _scan(
            directory: Path, extensions: set[str] | None = None
        ) -> list[dict[str, Any]]:
            if not directory.exists():
                return []
            files = []
            for p in sorted(directory.rglob("*")):
                if not p.is_file():
                    continue
                if p.name.startswith((".", "_")):
                    continue
                if extensions and p.suffix.lower() not in extensions:
                    continue
                stat = p.stat()
                try:
                    rel = p.relative_to(directory)
                except ValueError:
                    rel = Path(p.name)
                files.append({
                    "name": str(rel),
                    "size_kb": round(stat.st_size / 1024, 1),
                    "modified": datetime.fromtimestamp(
                        stat.st_mtime
                    ).strftime("%Y-%m-%d %H:%M"),
                })
            return files

        data_files = _scan(self.data_dir, {".csv", ".parquet"})
        model_files = _scan(self.models_dir, self._MODEL_EXTENSIONS)

        return {
            "data_files": data_files,
            "model_files": model_files,
            "data_dir": str(self.data_dir),
            "models_dir": str(self.models_dir),
        }

    def clear_cache(
        self, clear_data: bool = True, clear_models: bool = True
    ) -> dict[str, Any]:
        """Delete locally cached files.

        Args:
            clear_data: Remove files from ``data_dir``.
            clear_models: Remove files from ``models_dir``.

        Returns:
            Dict summarising what was deleted.
        """
        import shutil as _shutil

        removed: list[str] = []

        if clear_data and self.data_dir.exists():
            for p in sorted(self.data_dir.rglob("*"), reverse=True):
                if p.name.startswith("."):
                    continue
                if p.is_file():
                    p.unlink()
                    removed.append(str(p))
                elif p.is_dir() and not any(p.iterdir()):
                    p.rmdir()
            logger.info("Cleared {} data files", len(removed))

        models_before = len(removed)
        if clear_models and self.models_dir.exists():
            for p in self.models_dir.iterdir():
                if p.name.startswith("."):
                    continue
                if p.is_dir():
                    _shutil.rmtree(p)
                    # Count files that were inside
                    removed.append(str(p))
                elif p.is_file() and p.suffix.lower() in self._MODEL_EXTENSIONS:
                    p.unlink()
                    removed.append(str(p))
            logger.info(
                "Cleared {} model entries", len(removed) - models_before
            )

        return {"status": "ok", "removed": removed, "count": len(removed)}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_drive_id(url: str) -> str:
        """Extract the Google Drive file or folder ID from a URL.

        Handles:
          - ``https://drive.google.com/drive/folders/{id}``
          - ``https://drive.google.com/file/d/{id}/view``
          - ``https://drive.google.com/open?id={id}``
          - ``https://drive.google.com/uc?id={id}&export=download``
          - Raw ID string (no URL scheme)

        Args:
            url: Google Drive URL or raw ID.

        Returns:
            The extracted ID string.

        Raises:
            ValueError: If the URL format is not recognised.
        """
        url = url.strip()

        # Already a bare ID (no slashes, no dots)
        if re.fullmatch(r"[a-zA-Z0-9_-]{10,}", url):
            return url

        # /folders/{id}
        m = _FOLDER_RE.search(url)
        if m:
            return m.group(1)

        # /file/d/{id}
        m = _FILE_D_RE.search(url)
        if m:
            return m.group(1)

        # ?id={id} or &id={id}
        m = _OPEN_ID_RE.search(url)
        if m:
            return m.group(1)

        # Try query-string parsing as fallback
        parsed = urlparse(url)
        qs = parse_qs(parsed.query)
        if "id" in qs:
            return qs["id"][0]

        raise ValueError(
            f"Could not extract Google Drive ID from URL: {url!r}. "
            f"Expected a URL like https://drive.google.com/drive/folders/{{id}} "
            f"or https://drive.google.com/file/d/{{id}}/view"
        )
