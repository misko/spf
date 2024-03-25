def test_mavlink_radio_collect(script_runner):
    result = script_runner.run(
        [
            "./spf/mavlink_radio_collection.py",
            "--fake-drone",
            "--exit",
            "-c",
            "./tests/test_config.yaml",
            "-m",
            "./tests/test_device_mapping",
            "-r",
            "center",
            "-n",
            50,
        ]
    )

    assert result.returncode == 0
