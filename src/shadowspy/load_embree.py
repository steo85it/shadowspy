# ─── bootstrap embree-vars.sh in this Jupyter kernel ───
import os, subprocess, shlex, ctypes, glob

def source_env(script_path):
    """Source a bash script and capture its exported vars into os.environ."""
    # Use bash -lc so that 'source' works exactly like in your shell
    cmd = f"bash -lc 'source {shlex.quote(script_path)} && env'"
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True, executable="/bin/bash")
    for line in p.stdout:
        k, _, v = line.decode().partition("=")
        os.environ[k] = v.rstrip("\n")


# # 2) Now os.environ has CPATH, LIBRARY_PATH, DYLD_LIBRARY_PATH exactly as in your script
# print("CPATH:", os.environ.get("CPATH"))
# print("LIBRARY_PATH:", os.environ.get("LIBRARY_PATH"))
# print("DYLD_LIBRARY_PATH:", os.environ.get("DYLD_LIBRARY_PATH"))
#
# # 3) Preload Embree's shared library so Python extensions can see it
# #    We assume your embree-vars.sh sits next to 'lib/'.
# libdir = os.path.join(os.path.dirname(script), "lib")
# cands = glob.glob(os.path.join(libdir, "libembree.*"))
# if not cands:
#     raise RuntimeError(f"No libembree.* found in {libdir}")
# sofile = cands[0]
# ctypes.CDLL(sofile, mode=ctypes.RTLD_GLOBAL)
# print("Preloaded:", sofile)
#
# # 4) Now you can import any Python bindings that wrap Embree
# #    For example:
# import embree
# # print("Embree module loaded from", embree.__file__)
