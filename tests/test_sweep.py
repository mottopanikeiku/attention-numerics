import platform

from sweep import cpu_model


def test_cpu_model_reads_x86_model_name(tmp_path):
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text("processor\t: 0\nmodel name\t: Example CPU @ 3.00GHz\nflags\t: sse\n")
    assert cpu_model(cpuinfo) == "Example CPU @ 3.00GHz"


def test_cpu_model_falls_back_without_model_name(tmp_path):
    arm = tmp_path / "cpuinfo"
    arm.write_text("processor\t: 0\nCPU implementer\t: 0x41\nCPU part\t: 0xd40\n")
    expected = platform.processor() or platform.machine()
    assert cpu_model(arm) == expected
    assert cpu_model(tmp_path / "missing") == expected
