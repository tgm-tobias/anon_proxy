import atexit
import gc
import sys
import weakref

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


def _track_atexit(monkeypatch):
    """Make atexit.register/unregister append/remove from a list we can assert on."""
    registered = []
    monkeypatch.setattr(atexit, "register", lambda fn, *a, **k: registered.append(fn))
    monkeypatch.setattr(
        atexit,
        "unregister",
        lambda fn: registered.remove(fn) if fn in registered else None,
    )
    return registered


def test_atexit_handler_stops_running_child(monkeypatch):
    registered = _track_atexit(monkeypatch)
    sup = ProxySupervisor(cmd=[sys.executable, "-c", "import time; time.sleep(30)"])
    sup.start()
    assert sup.stop in registered  # registered once a child exists
    assert sup.is_running()

    for fn in list(registered):  # simulate interpreter exit on app Quit
        fn()

    assert not sup.is_running()  # child reaped, port released


def test_atexit_registration_tracks_child_lifecycle(monkeypatch):
    registered = _track_atexit(monkeypatch)
    sup = ProxySupervisor(cmd=[sys.executable, "-c", "import time; time.sleep(30)"])
    assert sup.stop not in registered  # construction registers nothing
    sup.start()
    assert sup.stop in registered  # a live child is registered for reaping
    sup.stop(grace=2.0)
    assert sup.stop not in registered  # reaped child unregisters


def test_restart_reregisters_atexit_cleanup(monkeypatch):
    registered = _track_atexit(monkeypatch)
    sup = ProxySupervisor(cmd=[sys.executable, "-c", "import time; time.sleep(30)"])
    sup.start()
    sup.restart()
    assert sup.stop in registered  # restarted child is still reaped at exit
    sup.stop(grace=2.0)
    assert sup.stop not in registered


def test_stopped_supervisor_is_garbage_collectable():
    # Uses the REAL atexit: after stop(), nothing pins the supervisor.
    sup = ProxySupervisor(cmd=[sys.executable, "-c", "import time; time.sleep(30)"])
    sup.start()
    sup.stop(grace=2.0)
    ref = weakref.ref(sup)
    del sup
    gc.collect()
    assert ref() is None  # not pinned by atexit -> collected


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
