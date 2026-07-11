"""Deterministic, strict source-subject discovery for production RAG."""

from __future__ import annotations

import stat
import unicodedata
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from src.config._rag_config import StrictRagConfigModel
from src.config.rag_index_config import CatalogConfig


class SubjectCatalogError(RuntimeError):
    """Base class for typed, fail-fast catalog errors."""

    def __init__(self, *, code: str, message: str, path: Path | None) -> None:
        self.code = code
        self.path = path
        super().__init__(f"{code}: {message}")


class CatalogRootError(SubjectCatalogError):
    """The configured data root is missing or not a directory."""


class CatalogPathError(SubjectCatalogError):
    """A discovered path is invalid, unreadable, or escapes the data root."""


class CatalogSymlinkError(SubjectCatalogError):
    """A symlink violates the explicitly configured symlink policy."""


class SubjectNormalizationCollisionError(SubjectCatalogError):
    """Two directory names normalize to the same subject identifier."""


class UnsupportedSubjectSourcesError(SubjectCatalogError):
    """A subject has files but none match the supported source contract."""


class SubjectPolicyMapError(SubjectCatalogError):
    """The discovered subject set differs from the policy-map subject set."""


class DuplicateSourceTargetError(SubjectCatalogError):
    """Multiple logical paths resolve to the same source file."""


class SourceCatalogEntry(StrictRagConfigModel):
    """One supported regular source file inside the configured data root."""

    subject_id: str
    source_path: Path
    source_relpath: str
    extension: str
    file_size_bytes: int


class SubjectCatalogEntry(StrictRagConfigModel):
    """One normalized subject and its deterministic source inventory."""

    subject_id: str
    directory_name: str
    directory_path: Path
    policy_id: str
    sources: tuple[SourceCatalogEntry, ...]


class SubjectCatalogSnapshot(StrictRagConfigModel):
    """Immutable result of one complete catalog discovery pass."""

    data_root: Path
    subjects: tuple[SubjectCatalogEntry, ...]
    subject_policy_map: dict[str, str]

    def subject_ids(self) -> tuple[str, ...]:
        return tuple(subject.subject_id for subject in self.subjects)

    def source_entries(self) -> tuple[SourceCatalogEntry, ...]:
        return tuple(source for subject in self.subjects for source in subject.sources)


def normalize_subject_id(
    value: str,
    normalization_version: Literal["subject_id_v1"],
) -> str:
    """Normalize a directory name using the one configured V1 algorithm."""
    if normalization_version != "subject_id_v1":
        raise ValueError("unsupported subject normalization version")
    normalized = unicodedata.normalize("NFKC", value).strip().casefold()
    output: list[str] = []
    for character in normalized:
        if character.isalnum() or character == "_":
            output.append(character)
        elif character.isspace() or character == "-":
            output.append("_")
    subject_id = "".join(output)
    while "__" in subject_id:
        subject_id = subject_id.replace("__", "_")
    return subject_id.strip("_")


class SubjectCatalog:
    """Discover supported sources and enforce an exact subject-policy map."""

    def __init__(
        self,
        *,
        config: CatalogConfig,
        subject_policy_map: Mapping[str, str],
    ) -> None:
        if not isinstance(config, CatalogConfig):
            raise TypeError("config must be a validated CatalogConfig")
        copied_map: dict[str, str] = {}
        for subject_id, policy_id in subject_policy_map.items():
            if not isinstance(subject_id, str) or not subject_id:
                raise SubjectPolicyMapError(
                    code="invalid_subject_policy_map",
                    message="subject-policy keys must be non-empty strings",
                    path=None,
                )
            if not isinstance(policy_id, str) or not policy_id:
                raise SubjectPolicyMapError(
                    code="invalid_subject_policy_map",
                    message="subject-policy values must be non-empty strings",
                    path=None,
                )
            copied_map[subject_id] = policy_id
        self._config = config
        self._subject_policy_map = copied_map

    def discover(self) -> SubjectCatalogSnapshot:
        """Perform a complete deterministic discovery or raise a typed error."""
        logical_root = self._config.data_root.absolute()
        try:
            resolved_root = self._config.data_root.resolve(strict=True)
        except OSError as exc:
            raise CatalogRootError(
                code="data_root_unreadable",
                message=f"data root cannot be resolved ({type(exc).__name__})",
                path=self._config.data_root,
            ) from exc
        if not resolved_root.is_dir():
            raise CatalogRootError(
                code="data_root_not_directory",
                message="configured data root is not a directory",
                path=self._config.data_root,
            )

        try:
            root_children = self._sorted_children(logical_root)
        except OSError as exc:
            raise CatalogRootError(
                code="data_root_unreadable",
                message=f"data root cannot be listed ({type(exc).__name__})",
                path=self._config.data_root,
            ) from exc

        directory_by_subject: dict[str, Path] = {}
        directory_name_by_subject: dict[str, str] = {}
        candidate_directories: list[tuple[str, str, Path]] = []
        for child in root_children:
            if self._is_excluded_name(child.name):
                continue
            resolved_child = self._resolve_checked(child, resolved_root)
            if not self._is_directory(resolved_child):
                continue
            subject_id = normalize_subject_id(
                child.name,
                self._config.normalization_version,
            )
            if not subject_id:
                raise CatalogPathError(
                    code="empty_normalized_subject",
                    message="subject directory normalizes to an empty identifier",
                    path=child,
                )
            if subject_id in directory_by_subject:
                previous_name = directory_name_by_subject[subject_id]
                raise SubjectNormalizationCollisionError(
                    code="subject_normalization_collision",
                    message=(
                        "multiple directory names normalize to subject "
                        f"'{subject_id}': '{previous_name}' and '{child.name}'"
                    ),
                    path=child,
                )
            directory_by_subject[subject_id] = child
            directory_name_by_subject[subject_id] = child.name
            candidate_directories.append((subject_id, child.name, child))

        seen_source_targets: dict[Path, str] = {}
        discovered: list[tuple[str, str, Path, tuple[SourceCatalogEntry, ...]]] = []
        for subject_id, directory_name, directory_path in candidate_directories:
            sources, regular_file_count = self._discover_subject_sources(
                subject_id=subject_id,
                subject_directory=directory_path,
                logical_root=logical_root,
                resolved_root=resolved_root,
                seen_source_targets=seen_source_targets,
                active_directories=set(),
            )
            if not sources:
                if regular_file_count:
                    raise UnsupportedSubjectSourcesError(
                        code="subject_has_no_supported_sources",
                        message=(
                            f"subject '{subject_id}' contains regular files but no "
                            "supported extensions"
                        ),
                        path=directory_path,
                    )
                continue
            discovered.append(
                (subject_id, directory_name, directory_path, tuple(sources))
            )

        discovered.sort(key=lambda item: item[0])
        discovered_ids = {subject_id for subject_id, _, _, _ in discovered}
        mapped_ids = set(self._subject_policy_map)
        if discovered_ids != mapped_ids:
            missing_policy = sorted(discovered_ids - mapped_ids)
            unknown_subject = sorted(mapped_ids - discovered_ids)
            raise SubjectPolicyMapError(
                code="subject_policy_map_mismatch",
                message=(
                    f"missing policy subjects={missing_policy}; "
                    f"unknown mapped subjects={unknown_subject}"
                ),
                path=logical_root,
            )

        subject_entries = tuple(
            SubjectCatalogEntry(
                subject_id=subject_id,
                directory_name=directory_name,
                directory_path=directory_path,
                policy_id=self._subject_policy_map[subject_id],
                sources=sources,
            )
            for subject_id, directory_name, directory_path, sources in discovered
        )
        ordered_policy_map = {
            subject.subject_id: subject.policy_id for subject in subject_entries
        }
        return SubjectCatalogSnapshot(
            data_root=resolved_root,
            subjects=subject_entries,
            subject_policy_map=ordered_policy_map,
        )

    def _discover_subject_sources(
        self,
        *,
        subject_id: str,
        subject_directory: Path,
        logical_root: Path,
        resolved_root: Path,
        seen_source_targets: dict[Path, str],
        active_directories: set[Path],
    ) -> tuple[list[SourceCatalogEntry], int]:
        resolved_directory = self._resolve_checked(subject_directory, resolved_root)
        if resolved_directory in active_directories:
            raise CatalogSymlinkError(
                code="symlink_directory_cycle",
                message="directory traversal encountered a cycle",
                path=subject_directory,
            )
        active_directories.add(resolved_directory)
        sources: list[SourceCatalogEntry] = []
        regular_file_count = 0
        try:
            children = self._sorted_children(subject_directory)
            for child in children:
                if self._is_excluded_name(child.name):
                    continue
                resolved_child = self._resolve_checked(child, resolved_root)
                if self._is_directory(resolved_child):
                    nested_sources, nested_regular_count = (
                        self._discover_subject_sources(
                            subject_id=subject_id,
                            subject_directory=child,
                            logical_root=logical_root,
                            resolved_root=resolved_root,
                            seen_source_targets=seen_source_targets,
                            active_directories=active_directories,
                        )
                    )
                    sources.extend(nested_sources)
                    regular_file_count += nested_regular_count
                    continue
                if not self._is_regular_file(resolved_child):
                    continue
                regular_file_count += 1
                extension = child.suffix.casefold()
                if extension not in self._config.supported_extensions:
                    continue
                source_relpath = child.relative_to(logical_root).as_posix()
                if resolved_child in seen_source_targets:
                    raise DuplicateSourceTargetError(
                        code="duplicate_source_target",
                        message=(
                            "multiple logical source paths resolve to the same file: "
                            f"'{seen_source_targets[resolved_child]}' and "
                            f"'{source_relpath}'"
                        ),
                        path=child,
                    )
                seen_source_targets[resolved_child] = source_relpath
                sources.append(
                    SourceCatalogEntry(
                        subject_id=subject_id,
                        source_path=resolved_child,
                        source_relpath=source_relpath,
                        extension=extension,
                        file_size_bytes=resolved_child.stat().st_size,
                    )
                )
        except SubjectCatalogError:
            raise
        except OSError as exc:
            raise CatalogPathError(
                code="source_tree_unreadable",
                message=f"source tree cannot be inspected ({type(exc).__name__})",
                path=subject_directory,
            ) from exc
        finally:
            active_directories.remove(resolved_directory)

        sources.sort(key=lambda entry: entry.source_relpath)
        return sources, regular_file_count

    def _resolve_checked(self, logical_path: Path, resolved_root: Path) -> Path:
        is_symlink = logical_path.is_symlink()
        try:
            resolved_path = logical_path.resolve(strict=True)
        except OSError as exc:
            raise CatalogPathError(
                code="path_unreadable",
                message=f"discovered path cannot be resolved ({type(exc).__name__})",
                path=logical_path,
            ) from exc
        if not resolved_path.is_relative_to(resolved_root):
            raise CatalogSymlinkError(
                code="symlink_escape",
                message="discovered path resolves outside data_root",
                path=logical_path,
            )
        if is_symlink and self._config.symlink_policy == "reject":
            raise CatalogSymlinkError(
                code="symlink_rejected",
                message="symlinks are disabled by catalog configuration",
                path=logical_path,
            )
        return resolved_path

    def _is_excluded_name(self, name: str) -> bool:
        if name in self._config.excluded_exact_names:
            return True
        if any(name.startswith(prefix) for prefix in self._config.excluded_prefixes):
            return True
        if self._config.exclude_hidden and name.startswith("."):
            return True
        if (
            self._config.exclude_cache_directories
            and name in self._config.cache_directory_names
        ):
            return True
        if (
            self._config.exclude_unclassified
            and name == self._config.unclassified_directory_name
        ):
            return True
        return bool(
            self._config.exclude_needs_ocr
            and name == self._config.needs_ocr_directory_name
        )

    @staticmethod
    def _sorted_children(directory: Path) -> list[Path]:
        return sorted(
            directory.iterdir(),
            key=lambda path: (path.name.casefold(), path.name),
        )

    @staticmethod
    def _is_directory(path: Path) -> bool:
        return stat.S_ISDIR(path.stat().st_mode)

    @staticmethod
    def _is_regular_file(path: Path) -> bool:
        return stat.S_ISREG(path.stat().st_mode)


__all__ = [
    "CatalogPathError",
    "CatalogRootError",
    "CatalogSymlinkError",
    "DuplicateSourceTargetError",
    "SourceCatalogEntry",
    "SubjectCatalog",
    "SubjectCatalogEntry",
    "SubjectCatalogError",
    "SubjectCatalogSnapshot",
    "SubjectNormalizationCollisionError",
    "SubjectPolicyMapError",
    "UnsupportedSubjectSourcesError",
    "normalize_subject_id",
]
