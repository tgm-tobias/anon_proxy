import sys

from anon_proxy.menubar.supervisor import (
    ProxySupervisor,
    install_launch_agent,
    launch_agent_plist,
    uninstall_launch_agent,
)


def test_start_stop_lifecycle():
    sup = ProxySupervisor(cmd=[sys.executable, "-c", "import time; time.sleep(30)"])
    assert sup.is_running() is False
    sup.start()
    assert sup.is_running() is True
    sup.stop(grace=2.0)
    assert sup.is_running() is False


def test_restart_replaces_process():
    sup = ProxySupervisor(cmd=[sys.executable, "-c", "import time; time.sleep(30)"])
    sup.start()
    first_pid = sup._proc.pid
    sup.restart()
    assert sup.is_running() is True
    assert sup._proc.pid != first_pid
    sup.stop(grace=2.0)


def test_start_is_idempotent_while_running():
    sup = ProxySupervisor(cmd=[sys.executable, "-c", "import time; time.sleep(30)"])
    sup.start()
    pid = sup._proc.pid
    sup.start()
    assert sup._proc.pid == pid
    sup.stop(grace=2.0)


def test_plist_contains_label_and_args():
    xml = launch_agent_plist(
        "com.anon-proxy.menubar",
        ["/usr/bin/env", "anon-proxy-menubar"],
        run_at_load=True,
    )
    assert "com.anon-proxy.menubar" in xml
    assert "anon-proxy-menubar" in xml
    assert "<key>RunAtLoad</key>" in xml
    assert xml.startswith("<?xml")


def test_install_and_uninstall_plist_file(tmp_path):
    path = install_launch_agent(
        "com.anon-proxy.menubar",
        ["anon-proxy-menubar"],
        plist_dir=tmp_path,
        load=False,
    )
    assert path.exists()
    assert path.name == "com.anon-proxy.menubar.plist"
    uninstall_launch_agent("com.anon-proxy.menubar", plist_dir=tmp_path, load=False)
    assert not path.exists()
