def test_foo_bar(script_runner):
    result = script_runner.run(
        [
            "./spf/mavlink_radio_collection.py",
            "--fake-drone",
            "--fake-radio",
            "--exit",
            "-c",
            "./tests/test_config.yaml",
            "-m",
            "./tests/test_device_mapping",
            "-r",
            "center",
        ]
    )

    assert result.returncode == 0
