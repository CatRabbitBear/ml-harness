from core.runtime.artifacts import build_run_artifact_dir, resolve_artifact_root


def test_resolve_artifact_root_uses_local_default_when_env_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("CORE_ARTIFACT_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)

    root = resolve_artifact_root()

    assert root == tmp_path / ".artifacts"
    assert root.exists()


def test_resolve_artifact_root_uses_env_var(tmp_path, monkeypatch):
    env_root = tmp_path / "custom_root"
    monkeypatch.setenv("CORE_ARTIFACT_ROOT", str(env_root))

    root = resolve_artifact_root()

    assert root == env_root
    assert root.exists()


def test_resolve_artifact_root_falls_back_when_env_invalid(tmp_path, monkeypatch):
    invalid_root = tmp_path / "not_a_dir"
    invalid_root.write_text("nope", encoding="utf-8")
    monkeypatch.setenv("CORE_ARTIFACT_ROOT", str(invalid_root))
    monkeypatch.chdir(tmp_path)

    root = resolve_artifact_root()

    assert root == tmp_path / ".artifacts"
    assert root.exists()


def test_build_run_artifact_dir_creates_run_dir(tmp_path):
    run_dir = build_run_artifact_dir("run_123", artifact_root=tmp_path)

    assert run_dir == tmp_path / "runs" / "run_123"
    assert run_dir.exists()


def test_build_run_artifact_dir_uses_default_root(tmp_path, monkeypatch):
    monkeypatch.delenv("CORE_ARTIFACT_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)

    run_dir = build_run_artifact_dir("run_456")

    assert run_dir == tmp_path / ".artifacts" / "runs" / "run_456"
    assert run_dir.exists()
