import os
import signal
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import CubicSpline

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Geometry conversion
# ─────────────────────────────────────────────────────────────────────────────



def vector_to_xfoil_dat(vector, filename="temp_airfoil.dat", n_points=150):
    """
    Converts a flattened coordinate vector into a Selig-formatted .dat file.

    Strategy:
    - Cosine x-redistribution to cluster points near LE/TE
    - CubicSpline with clamped LE (dy/dx=0 at x=0) to eliminate the cusp
      WITHOUT an aggressive blend ramp that creates kinks
    - Exact pinning at LE (0,0) and TE closure
    """
    half = len(vector) // 2
    x_linear = np.linspace(0, 1, half)
    y_upper   = vector[:half]
    y_lower   = vector[half:]

    # ── 1. Cosine x-distribution ─────────────────────────────────────────────
    theta = np.linspace(0, np.pi, n_points)
    x_cos = (1 - np.cos(theta)) / 2        # clusters near x=0 and x=1

    # ── 2. Fit splines with clamped slope at LE (bc_type forces dy/dx=0 at x=0)
    # This directly kills the cusp without any post-hoc blending that causes kinks
    cs_upper = CubicSpline(
        x_linear, y_upper,
        bc_type=((1, 0.0), "not-a-knot")   # dy/dx = 0 at x=0 (LE), free at TE
    )
    cs_lower = CubicSpline(
        x_linear, y_lower,
        bc_type=((1, 0.0), "not-a-knot")   # dy/dx = 0 at x=0 (LE), free at TE
    )

    y_upper_cos = cs_upper(x_cos)
    y_lower_cos = cs_lower(x_cos)

    # ── 3. Pin LE exactly at (0, 0) ──────────────────────────────────────────
    y_upper_cos[0] = 0.0
    y_lower_cos[0] = 0.0

    # ── 4. Close TE: average any gap at x=1 ──────────────────────────────────
    te_avg           = (y_upper_cos[-1] + y_lower_cos[-1]) / 2
    y_upper_cos[-1]  = te_avg
    y_lower_cos[-1]  = te_avg

    # ── 5. Sanity check: upper must stay above lower everywhere ───────────────
    crossed = np.where(y_upper_cos < y_lower_cos)[0]
    if len(crossed) > 0:
        # Swap crossed points — indicates degenerate input geometry
        for i in crossed:
            y_upper_cos[i], y_lower_cos[i] = y_lower_cos[i], y_upper_cos[i]

    # ── 6. Build Selig format: TE→LE (upper), LE→TE (lower) ──────────────────
    x_selig = np.concatenate([x_cos[::-1], x_cos[1:]])
    y_selig = np.concatenate([y_upper_cos[::-1], y_lower_cos[1:]])

    # ── 7. Write .dat ─────────────────────────────────────────────────────────
    with open(filename, "w") as f:
        f.write("Generated_Airfoil\n")
        for x, y in zip(x_selig, y_selig):
            f.write(f"{x:.8f} {y:.8f}\n")

    return filename


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — XFOIL runner (single airfoil)
# ─────────────────────────────────────────────────────────────────────────────

def _run_xfoil_polar(Re, dat_file, polar_file, a_min=-5, a_max=15, a_step=0.5, timeout=60):

    commands = [
        # ── Graphics off ───────────────────────────────────────────────────
        "PLOP", "G", "",

        # ── Load + normalize ───────────────────────────────────────────────
        f"LOAD {dat_file}", "",
        "NORM",
        "PCOP",
        "PANE",

        # ── Increase panel count + LE bunching ─────────────────────────────
        "PPAR",
        "N", "260",
        "P", "3",
        "",

        # ── OPER: inviscid pass first to initialize BL cleanly ─────────────
        "OPER",

        # Inviscid at α=0 — seeds a clean Cp distribution with no BL
        "A 0",

        # Now switch to viscous — BL initializes from a converged inviscid field
        # instead of from scratch, which is what causes the MRCHDU spiral
        f"VISC {Re}",
        "ITER 200",

        # Tighten BL solver: reduce arc-length step + lower Ncrit
        "VPAR",
        "VAR 0.3",      # Much smaller step (default=1.0): slow but stable
        "N 5",          # Low Ncrit: trips transition early, avoids laminar separation bubble
        "",

        # ── Polar accumulation ─────────────────────────────────────────────
        "PACC",
        polar_file,
        "",

        # Re-solve at α=0 viscously as the seed point
        "INIT",         # Reset BL — critical after switching from inviscid
        "A 0",

        # Sweep positive then negative from the converged seed
        f"ASEQ {a_step} {a_max} {a_step}",
        f"ASEQ -{a_step} {a_min} -{a_step}",

        "PACC", "",
        "QUIT",
    ]

    command_string = "\n".join(commands) + "\n"

    process = None
    try:
        process = subprocess.Popen(
            ["xvfb-run", "--auto-servernum", "xfoil"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setsid,
        )
        process.communicate(input=command_string, timeout=timeout)
        return True

    except subprocess.TimeoutExpired:
        if process is not None:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                process.communicate(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                process.communicate()
            except ProcessLookupError:
                pass
        return False

    except Exception as e:
        if process is not None:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                process.communicate()
            except Exception:
                pass
        raise RuntimeError(f"XFOIL Popen failed: {e}") from e


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Polar parser
# ─────────────────────────────────────────────────────────────────────────────

def _parse_polar(polar_file):
    if not os.path.exists(polar_file):
        return None
    try:
        with open(polar_file) as f:
            lines = f.readlines()

        # Find data start dynamically
        skip = 0
        for i, line in enumerate(lines):
            if line.strip().startswith("------"):
                skip = i + 1
                break
        else:
            return None

        df = pd.read_csv(
            polar_file,
            skiprows=skip,
            sep=r"\s+",
            names=["alpha", "CL", "CD", "CDp", "CM", "Top_Xtr", "Bot_Xtr",
                   "Top_Itr", "Bot_Itr"],  # XFOIL 6.99 writes 9 columns not 7
            engine="python",
        )

        if df.empty:
            return None

        # Drop unphysical rows only — keep partially converged data
        df = df[
            (df["CD"] > 0) &          # Negative drag = non-converged
            (df["CD"] < 0.5) &        # CD > 0.5 is certainly garbage
            (df["CL"].abs() < 5.0)    # |CL| > 5 is certainly garbage
        ].reset_index(drop=True)

        return df[["alpha", "CL", "CD"]].copy() if not df.empty else None

    except Exception as e:
        print(f"  [parse] Exception: {e}")
        return None

    finally:
        if os.path.exists(polar_file):
            os.remove(polar_file)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — MASTER FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def run_polar_from_npy(
    npy_path,
    Re,
    a_min=-5,
    a_max=15,
    a_step=0.5,
    timeout=60,
    verbose=True,
):
    """
    Master pipeline: .npy matrix → per-airfoil XFOIL polars.

    Parameters
    ----------
    npy_path : str | Path
        Path to the .npy file. Shape: (N_airfoils, N_coords).
        Each row is a flattened [y_upper..., y_lower...] vector.
    Re : float
        Reynolds number for viscous analysis.
    a_min, a_max, a_step : float
        Angle-of-attack sweep range and increment (degrees).
    timeout : int
        Per-airfoil XFOIL timeout in seconds.
    verbose : bool
        Print per-airfoil status.

    Returns
    -------
    results : list of dict
        One entry per airfoil:
        {
            "index":  int,               # row index in the .npy matrix
            "polar":  pd.DataFrame | None,  # alpha / CL / CD  (None = failed)
            "status": "ok" | "xfoil_failed" | "parse_failed" | "timeout"
        }
    """
    npy_path   = Path(npy_path)
    dat_file   = "temp_airfoil.dat"
    polar_file = "temp_polar.txt"

    # ── Load the coordinate matrix ──────────────────────────────────────────
    coords_matrix = np.load(npy_path)                   # shape (N, D)
    if coords_matrix.ndim == 1:                         # single airfoil edge case
        coords_matrix = coords_matrix[np.newaxis, :]
    n_airfoils = len(coords_matrix)

    if verbose:
        print(f"Loaded {n_airfoils} airfoil(s) from {npy_path.name}")
        print(f"Re={Re:.2e}  α=[{a_min}°..{a_max}°]  step={a_step}°\n")

    results = []

    for i, vector in enumerate(coords_matrix):

        # ── Cleanup from any previous iteration ─────────────────────────────
        for f in (dat_file, polar_file):
            if os.path.exists(f):
                os.remove(f)

        status = "ok"
        polar  = None

        try:
            # STEP A: vector → .dat file
            vector_to_xfoil_dat(vector, filename=dat_file)

            # STEP B: .dat → XFOIL → polar file
            success = _run_xfoil_polar(
                Re, dat_file, polar_file,
                a_min=a_min, a_max=a_max,
                a_step=a_step, timeout=timeout,
            )

            if not success:
                status = "timeout"
            else:
                # STEP C: polar file → DataFrame
                polar = _parse_polar(polar_file)
                if polar is None:
                    status = "parse_failed"

        except Exception as exc:
            status = "xfoil_failed"
            if verbose:
                print(f"  [airfoil {i:>4d}] ERROR — {exc}")

        finally:
            # STEP D: always delete temp files before next iteration
            for f in (dat_file, polar_file):
                if os.path.exists(f):
                    os.remove(f)

        results.append({"index": i, "polar": polar, "status": status})

        if verbose:
            n_pts = len(polar) if polar is not None else 0
            print(f"  [airfoil {i:>4d}/{n_airfoils}]  status={status:<14s}  "
                  f"converged_points={n_pts}")

    # ── Summary ─────────────────────────────────────────────────────────────
    if verbose:
        ok      = sum(1 for r in results if r["status"] == "ok")
        failed  = n_airfoils - ok
        print(f"\nDone — {ok}/{n_airfoils} succeeded, {failed} failed.")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Example usage
# ─────────────────────────────────────────────────────────────────────────────

#if __name__ == "__main__":
    results = run_polar_from_npy(
        npy_path="coords_matrix.npy",
        Re=1e6,
        a_min=-5,
        a_max=15,
        a_step=0.5,
        timeout=60,
        verbose=True,
    )

    # Collect only successful polars
    polars = {r["index"]: r["polar"] for r in results if r["status"] == "ok"}
    print(f"\nSuccessfully retrieved {len(polars)} polars.")


#-----------------------------------------

def debug_single_airfoil(vector, Re, index=0):
    """
    Runs ONE airfoil with full stdout capture and prints everything.
    Call this on your first failing airfoil to see what XFOIL actually says.
    """
    dat_file   = f"debug_airfoil_{index}.dat"
    polar_file = f"debug_polar_{index}.txt"

    # Write the .dat file
    vector_to_xfoil_dat(vector, filename=dat_file)
    print(f"=== .dat file written: {dat_file} ===")
    with open(dat_file) as f:
        print(f.read())

    commands = [
        "PLOP", "G", "",
        f"LOAD {dat_file}", "",
        "NORM",
        "PCOP",
        "PANE",
        "OPER",
        f"VISC {Re}",
        "ITER 200",
        "PACC",
        polar_file,
        "",
        "A 0",
        f"ASEQ 0.5 15 0.5",
        f"ASEQ -0.5 -5 -0.5",
        "PACC", "",
        "QUIT",
    ]
    command_string = "\n".join(commands) + "\n"

    process = subprocess.Popen(
    # fedisableexcept disables the FPE trap at the process level
    # This lets XFOIL produce NaN/inf instead of crashing on bad BL states
    ["bash", "-c",
     "python3 -c 'import ctypes; ctypes.CDLL(None).fedisableexcept(248)' 2>/dev/null; "
     "xvfb-run --auto-servernum xfoil"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    preexec_fn=os.setsid,
    )
    stdout, stderr = process.communicate(input=command_string, timeout=60)

    print("=== XFOIL STDOUT ===")
    print(stdout)
    print("=== XFOIL STDERR ===")
    print(stderr)

    if os.path.exists(polar_file):
        print(f"=== POLAR FILE ({polar_file}) ===")
        with open(polar_file) as f:
            print(f.read())
    else:
        print("=== POLAR FILE: NOT CREATED ===")

# Run it:
coords = np.load("coords_matrix.npy")
debug_single_airfoil(coords[0], Re=1e6)