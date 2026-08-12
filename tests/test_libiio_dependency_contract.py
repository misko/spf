from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_python_dependencies_match_hardware_qualified_stack():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert "pyadi-iio==0.0.18" in project["dependencies"]
    assert "pylibiio>=0.25,<0.27" in project["dependencies"]


def test_installer_pins_both_supported_libiio_source_commits():
    installer = (ROOT / "install_spf_libiio.sh").read_text()
    assert "spf-frame-metadata-source/v0.25-final-v3" in installer
    assert "c26258bfa33098c2b215e19cf85d448e89499b1a" in installer
    assert "spf-frame-metadata-source/v0.26-final-v3" in installer
    assert "d5695c3eaa9cec99cc6f7b2c91565555044b907a" in installer
    assert 'hasattr(iio, "MetadataBuffer")' in installer


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
