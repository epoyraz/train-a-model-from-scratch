"""Make MSVC's cl.exe importable into the current process environment on Windows.

torch.compile + Triton need a C compiler to build CUDA driver shims. On Windows
that must be MSVC (cl.exe); the mingw gcc that ships on many PATHs fails to link
against the CPython import library. Visual Studio / Build Tools installs cl.exe
but only exposes it inside a "Developer" shell that has run vcvars64.bat.

This helper locates vcvars64.bat (via vswhere), runs it, and copies the
environment variables it sets (PATH, INCLUDE, LIB, LIBPATH, ...) into os.environ
so child processes spawned by Triton can find cl.exe. No-op off Windows or when
cl.exe is already reachable.
"""
import os
import shutil
import subprocess


def _find_vcvars():
    program_files_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    vswhere = os.path.join(program_files_x86, "Microsoft Visual Studio", "Installer", "vswhere.exe")
    if os.path.exists(vswhere):
        try:
            install_path = subprocess.check_output(
                [vswhere, "-latest", "-products", "*",
                 "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                 "-property", "installationPath"],
                text=True,
            ).strip().splitlines()
            if install_path:
                candidate = os.path.join(install_path[0], "VC", "Auxiliary", "Build", "vcvars64.bat")
                if os.path.exists(candidate):
                    return candidate
        except (subprocess.CalledProcessError, OSError):
            pass
    # Fallback: probe the common install locations directly.
    for edition in ("BuildTools", "Community", "Professional", "Enterprise"):
        for year in ("2022", "2019"):
            base = "Program Files" if year == "2022" else "Program Files (x86)"
            candidate = os.path.join(
                "C:\\", base, "Microsoft Visual Studio", year, edition,
                "VC", "Auxiliary", "Build", "vcvars64.bat",
            )
            if os.path.exists(candidate):
                return candidate
    return None


def ensure_msvc_env(verbose=True):
    """Best-effort: put MSVC on PATH for this process. Returns True on success."""
    if os.name != "nt":
        return True
    if shutil.which("cl"):
        return True

    vcvars = _find_vcvars()
    if not vcvars:
        if verbose:
            print("[msvc_env] cl.exe not found and no Visual Studio Build Tools detected.")
            print("           torch.compile will fall back / fail. Install VS Build Tools with")
            print("           the 'Desktop development with C++' workload, or run from the")
            print("           'x64 Native Tools Command Prompt for VS'.")
        return False

    try:
        # Run vcvars64.bat then dump the resulting environment.
        out = subprocess.check_output(
            f'cmd /c ""{vcvars}" >nul 2>&1 && set"',
            text=True, shell=True,
        )
    except (subprocess.CalledProcessError, OSError) as e:
        if verbose:
            print(f"[msvc_env] failed to run vcvars64.bat: {e}")
        return False

    for line in out.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            os.environ[key] = value

    if shutil.which("cl"):
        if verbose:
            print(f"[msvc_env] MSVC ready: {shutil.which('cl')}")
        return True
    if verbose:
        print("[msvc_env] sourced vcvars but cl.exe still not found.")
    return False


if __name__ == "__main__":
    ok = ensure_msvc_env()
    print("MSVC available:", ok, "->", shutil.which("cl"))
