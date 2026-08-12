from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_python_dependencies_match_hardware_qualified_stack():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert "pyadi-iio==0.0.18" in project["dependencies"]
    assert "pylibiio>=0.25,<0.27" in project["dependencies"]


def test_installer_pins_both_supported_libiio_source_commits():
    versions = (ROOT / "packaging/libiio/versions.sh").read_text()
    assert "spf-frame-metadata-source/v0.25-final-v3" in versions
    assert "c26258bfa33098c2b215e19cf85d448e89499b1a" in versions
    assert "spf-frame-metadata-source/v0.26-final-v3" in versions
    assert "d5695c3eaa9cec99cc6f7b2c91565555044b907a" in versions

    installer = (ROOT / "install_spf_libiio.sh").read_text()
    assert "packaging/libiio/versions.sh" in installer
    assert 'hasattr(iio, "MetadataBuffer")' in installer


def test_binary_artifact_workflow_covers_pi_and_x86_64():
    workflow = (ROOT / ".github/workflows/libiio-packages.yml").read_text()
    assert "ubuntu-24.04-arm" in workflow
    assert "architecture: arm64" in workflow
    assert "architecture: amd64" in workflow
    assert "container: debian:12" in workflow
    assert "packaging/libiio/test_artifacts.sh" in workflow
    assert "python -m pytest" in workflow
    assert "tests/test_libiio_dependency_contract.py" in workflow
    assert "python3 -m pytest" not in workflow

    legacy_workflow = (ROOT / ".github/workflows/docker-build-and-test.yml").read_text()
    assert '"packaging/libiio/**"' in legacy_workflow
    assert '".github/workflows/libiio-packages.yml"' in legacy_workflow


def test_binary_installer_verifies_bundle_before_installing():
    installer = (ROOT / "install_spf_libiio_artifacts.sh").read_text()
    checksum_offset = installer.index("sha256sum --check SHA256SUMS")
    apt_offset = installer.index("apt-get install")
    pip_offset = installer.index('"$python_bin" -m pip install')
    assert checksum_offset < apt_offset < pip_offset
    assert 'hasattr(iio, "MetadataBuffer")' in installer

    builder = (ROOT / "packaging/libiio/build_artifacts.sh").read_text()
    assert "cross_compiling = True" in builder
    assert "matching staged .deb is intentionally not installed" in builder


def test_legacy_sdr_requirements_do_not_downgrade_pyadi():
    requirements = (ROOT / "spf/sdrpluto/requirements.txt").read_text().splitlines()
    assert "pyadi-iio==0.0.18" in requirements
    assert "pylibiio>=0.25,<0.27" in requirements


def test_rover_provisioning_installs_patched_binding_after_pip():
    provisioning = (
        ROOT / "data_collection/rover/rover_v3.1/provision_rover.sh"
    ).read_text()
    pip_offset = provisioning.index('"${VENV}/bin/pip" -q install -e')
    libiio_offset = provisioning.index('"${REPO_ROOT}/install_spf_libiio.sh"')
    assert libiio_offset > pip_offset
