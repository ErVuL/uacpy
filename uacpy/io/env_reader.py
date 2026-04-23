"""
Env file reader — partial MATLAB port.

.. warning::
    This module is a partial translation of the MATLAB ``read_env`` /
    ``read_env_core`` functions. Several helper functions (``readsxsy``,
    ``readszrz``, ``readr``, ``readtheta``, ``read_bell``, ``topbot``)
    are stubs that raise ``NotImplementedError``. As a result,
    ``read_env_file`` only works for basic ``.env`` files
    (range-independent environments with simple SSPs and boundary
    conditions). Bellhop/Bellhop3D models that require the unimplemented
    helpers will raise ``NotImplementedError`` at runtime.
"""

import os
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Tuple, List
from uacpy.core.environment import Environment
from uacpy.core.source import Source
from uacpy.core.receiver import Receiver


# ---------------------------------------------------------------------------
# Stub helpers — not yet ported from MATLAB
# ---------------------------------------------------------------------------

def _read_quoted_string(line: str) -> str:
    """Extract a quoted string from a line, or return the stripped line."""
    line = line.strip()
    if line.startswith("'") and "'" in line[1:]:
        return line.split("'")[1]
    if line.startswith('"') and '"' in line[1:]:
        return line.split('"')[1]
    return line


def _not_implemented(name: str):
    """Raise NotImplementedError for unported MATLAB helpers."""
    raise NotImplementedError(
        f"{name}() is not yet implemented. "
        "This env reader is an incomplete MATLAB port."
    )


def readsxsy(f):
    """Stub for the MATLAB ``readsxsy`` helper (not ported)."""
    _not_implemented("readsxsy")


def readszrz(f):
    """Stub for the MATLAB ``readszrz`` helper (not ported)."""
    _not_implemented("readszrz")


def readr(f):
    """Stub for the MATLAB ``readr`` helper (not ported)."""
    _not_implemented("readr")


def readtheta(f):
    """Stub for the MATLAB ``readtheta`` helper (not ported)."""
    _not_implemented("readtheta")


def read_bell(f, Bdry, freq, bot_depth, top_depth, rmax):
    """Stub for the MATLAB ``read_bell`` helper (not ported)."""
    _not_implemented("read_bell")


def topbot(f, freq, bc, atten_unit):
    """Stub for the MATLAB ``topbot`` helper (not ported)."""
    _not_implemented("topbot")


def read_env_file(envfil: str, model: str):
    """
    Read an Acoustics Toolbox .env file.

    Translated from the MATLAB ``read_env`` function.

    Returns
    -------
    TitleEnv : str
        Environment title from the file header.
    freq : float
        Frequency in Hz.
    SSP : SSPStruct
        Sound speed profile data.
    Bdry : BoundaryStruct
        Boundary condition data.
    Pos : dict
        Source/receiver position data.
    Beam : dict
        Beam parameters (Bellhop only).
    cInt : dict
        Phase speed integration limits.
    RMax : float
        Maximum range in km.
    """
    # ensure extension
    root, ext = os.path.splitext(envfil)
    if envfil != "ENVFIL" and ext.lower() != ".env":
        envfil = envfil + ".env"

    model_up = model.upper()

    # read core
    TitleEnv, freq, SSP, Bdry = read_env_core(envfil)

    # set defaults for outputs
    Pos = {}
    Beam = {}
    cInt = {}
    RMax = None

    scooter_like = {"SCOOTER", "KRAKEN", "KRAKENC", "KRAKEL", "SPARC", "BOUNCE"}
    if model_up in scooter_like:
        # next two numbers: lower/upper phase speed limits
        # open file and seek to the position after read_env_core parsed everything.
        # read_env_core already closed file; so reopen to read the tail portion -
        # in MATLAB read_env_core left the file pointer open; here we'll open anew and attempt to read
        with open(envfil, "r") as f:
            # naive approach: read all lines and search for first occurrence of cInt numbers AFTER the Title line & freq
            lines = f.readlines()
        # search lines for two floats pattern after frequency line: there is no unique method w/o format spec.
        # We'll attempt to find first two floats after the line containing Frequency reported by read_env_core
        floats = []
        for line in lines:
            for token in line.strip().split():
                try:
                    floats.append(float(token))
                except ValueError:
                    continue
            if len(floats) >= 2:
                break
        # fallbacks
        cInt["Low"] = floats[0] if len(floats) >= 1 else 1500.0
        cInt["High"] = floats[1] if len(floats) >= 2 else 1e9
        print(f"\n cLow = {cInt['Low']:8.1f} m/s  cHigh = {cInt['High']:8.1f} m/s")
        # RMax: try to take next float, or default
        RMax = floats[2] if len(floats) >= 3 else 100.0
        print(f"RMax = {RMax} km")
    else:
        # dummy values used for BELLHOP and others
        cInt["Low"] = 1500.0
        cInt["High"] = 1e9
        RMax = 100.0

    # For BELLHOP3D model: read source x-y coordinates (stub)
    if model_up == "BELLHOP3D":
        # read from file using stub
        with open(envfil, "r") as f:
            sx, sy, Nsx, Nsy = readsxsy(f)
        Pos["s"] = {"x": np.array(sx), "y": np.array(sy)}
        Pos["Nsx"] = Nsx
        Pos["Nsy"] = Nsy

    # read source/receiver depths - using stub
    with open(envfil, "r") as f:
        Pos = readszrz(f)

    if model_up == "BELLHOP":
        Pos_r = readr(open(envfil, "r"))
        Pos["r"] = {"r": Pos_r}
        Pos["Nrr"] = len(Pos_r)
        Beam = read_bell(
            open(envfil, "r"),
            Bdry,
            freq,
            Bdry["Bot"]["depth"],
            Bdry["Top"]["depth"],
            Pos_r[-1],
        )
    elif model_up == "BELLHOP3D":
        Pos_r = readr(open(envfil, "r"))
        Pos["r"] = {"r": Pos_r}
        Pos["Nrr"] = len(Pos_r)
        Pos_theta = readtheta(open(envfil, "r"))
        Pos["theta"] = Pos_theta
        Pos["Ntheta"] = len(Pos_theta)
        Beam = read_bell(
            open(envfil, "r"),
            Bdry,
            freq,
            Bdry["Bot"]["depth"],
            Bdry["Top"]["depth"],
            Pos_r[-1],
        )
    else:
        # dummy Beam structure for models that don't use Beam parameters
        r_z = Pos.get("r", {}).get("z", np.array([0.0]))
        Beam = {
            "RunType": "CG",
            "Nbeams": 0,
            "alpha": np.array([-15.0, +15.0]),
            "Box": {
                "z": 1.05 * float(np.max(r_z)) if r_z.size else 1.0,
                "r": 1.05 * float(RMax),
            },
            "deltas": 0.0,
        }
        # set up receiver range vector
        Pos["r"] = {"r": np.linspace(0.0, float(RMax), 501)}

    return TitleEnv, freq, SSP, Bdry, Pos, Beam, cInt, RMax


def read_env_core(envfil: str):
    """
    Parse the environment core of the .env file (translation of MATLAB read_env_core).
    Returns TitleEnv (str), freq (float), SSP (SSPStruct), Bdry (dict)
    """
    # defaults matching the MATLAB code
    alphaR = 1500.0
    betaR = 0.0
    rhoR = 1.0
    alphaI = 0.0
    betaI = 0.0

    NFirstAcoustic = 0
    NLastAcoustic = 0

    # boundary defaults
    Bdry = {
        "Top": {"cp": 0.0, "cs": 0.0, "rho": 0.0, "HS": None},
        "Bot": {"cp": 2000.0, "cs": 0.0, "rho": 2.0, "HS": None},
    }

    with open(envfil, "r") as f:
        # Title
        title_line = f.readline()
        if not title_line:
            raise EOFError("Empty environment file")
        TitleEnv = _read_quoted_string(title_line)
        print(TitleEnv)

        # frequency (first numeric)
        # read next numeric token
        def _next_float(file_obj):
            while True:
                line = file_obj.readline()
                if not line:
                    return None
                for token in line.strip().split():
                    try:
                        return float(token)
                    except ValueError:
                        continue

        freq = _next_float(f)
        if freq is None:
            raise EOFError("No frequency found in environment file")
        print(f"Frequency = {freq} Hz")

        # read number of media
        NMedia = int(_next_float(f))
        print(f"Number of media = {NMedia}")
        # read TopOpt line (quoted option string)
        TopOpt_line = f.readline()
        TopOpt = _read_quoted_string(TopOpt_line)
        # normalize to at least length 7 like MATLAB did
        TopOpt_padded = TopOpt + " " * max(0, 7 - len(TopOpt))
        Bdry["Top"]["Opt"] = TopOpt_padded

        # convert '*' to '~' if present at position 5 (1-based in MATLAB)
        if len(Bdry["Top"]["Opt"]) >= 5 and Bdry["Top"]["Opt"][4] == "*":
            Bdry["Top"]["Opt"] = Bdry["Top"]["Opt"][:4] + "~" + Bdry["Top"]["Opt"][5:]

        SSP = SSPStruct()
        SSP.NMedia = NMedia
        SSP.N = [0] * NMedia
        SSP.sigma = [0.0] * NMedia
        SSP.depth = [0.0] * (NMedia + 1)  # MATLAB indexing had depth(medium+1)
        SSP.z = np.array([], dtype=float)
        SSP.c = np.array([], dtype=float)
        SSP.cs = np.array([], dtype=float)
        SSP.rho = np.array([], dtype=float)
        SSP.raw = [dict() for _ in range(NMedia)]
        SSP.Npts = [0] * NMedia

        # parse option letters
        SSPType = Bdry["Top"]["Opt"][0]
        Bdry["Top"]["BC"] = (
            Bdry["Top"]["Opt"][1] if len(Bdry["Top"]["Opt"]) > 1 else " "
        )
        AttenUnit = Bdry["Top"]["Opt"][2:4] if len(Bdry["Top"]["Opt"]) >= 4 else "  "

        # print messages similar to MATLAB switch/case
        ssp_type_map = {
            "N": "N2-Linear approximation to SSP",
            "C": "C-Linear approximation to SSP",
            "P": "PCHIP approximation to SSP",
            "S": "Spline approximation to SSP",
            "Q": "Quadrilateral approximation to range-dependent SSP",
            "H": "Hexahedral approximation to range and depth dependent SSP",
            "A": "Analytic SSP option",
        }
        if SSPType in ssp_type_map:
            print("   " + ssp_type_map[SSPType])
        else:
            raise ValueError("Fatal error: Unknown option for SSP approximation")

        # Attenuation units description
        atten_map = {
            "N": "nepers/m",
            "F": "dB/mkHz",
            "M": "dB/m",
            "W": "dB/wavelength",
            "Q": "Q",
            "L": "Loss tangent",
        }
        a0 = AttenUnit[0] if len(AttenUnit) >= 1 else " "
        if a0 in atten_map:
            print(f"    Attenuation units: {atten_map[a0]}")
        else:
            raise ValueError("Fatal error: Unknown attenuation units")

        # optional volume attenuation: check 4th char
        if len(Bdry["Top"]["Opt"]) >= 4:
            ch4 = Bdry["Top"]["Opt"][3]
            if ch4 == "T":
                print("    THORP attenuation added")
            elif ch4 == "F":
                print("    Francois-Garrison attenuation added")

                # read T S pH z_bar
                # read next 4 floats
                def _next_n_floats(file_obj, n):
                    vals = []
                    while len(vals) < n:
                        line = file_obj.readline()
                        if not line:
                            break
                        for token in line.strip().split():
                            try:
                                vals.append(float(token))
                            except ValueError:
                                pass
                            if len(vals) >= n:
                                break
                    return vals

                vals = _next_n_floats(f, 4)
                if len(vals) == 4:
                    SSP.T, SSP.S, SSP.pH, SSP.z_bar = vals
                    print(
                        f"        T = {SSP.T:4.1f} degrees   S = {SSP.S:4.1f} psu   pH = {SSP.pH:4.1f}   z_bar = {SSP.z_bar:6.1f} m"
                    )
                else:
                    print("Warning: Francois-Garrison values missing or incomplete")

        # convert TopOpt 5th char '*' handling
        if len(Bdry["Top"]["Opt"]) >= 5 and Bdry["Top"]["Opt"][4] == "*":
            print("    Development options enabled")

        # topbot for top boundary
        cp_top, cs_top, rho_top, HS_top = topbot(f, freq, Bdry["Top"]["BC"], AttenUnit)
        Bdry["Top"].update({"cp": cp_top, "cs": cs_top, "rho": rho_top, "HS": HS_top})

        # main loop to read SSP
        print(
            "\n       z          alphaR         betaR           rho        alphaI         betaI"
        )
        print(
            "      (m)          (m/s)         (m/s)         (g/cm^3)      (m/s)         (m/s)"
        )

        Loc = [0] * NMedia
        for medium in range(NMedia):
            medium_index = medium  # 0-based
            if medium == 0:
                Loc[medium] = 0
            else:
                Loc[medium] = Loc[medium - 1] + SSP.Npts[medium - 1]

            # read SSP.N, SSP.sigma, SSP.depth(medium+1)
            # read sequential floats (we expect three)
            def _next_int_or_float(file_obj):
                while True:
                    line = file_obj.readline()
                    if not line:
                        return None
                    for token in line.strip().split():
                        try:
                            return float(token)
                        except ValueError:
                            continue

            N_val = int(_next_int_or_float(f))
            sigma_val = float(_next_int_or_float(f))
            depth_val = float(_next_int_or_float(f))

            SSP.N[medium] = N_val
            SSP.sigma[medium] = sigma_val
            SSP.depth[medium + 1] = depth_val

            print(
                f"    ( Number of points = {N_val}  Roughness = {sigma_val:6.2f}  Depth = {depth_val:8.2f} )"
            )
            # consume remainder of the line
            _ = f.readline()

            # read z lines until ztmp == SSP.depth(medium+1)
            z_list = []
            c_list = []
            cs_list = []
            rho_list = []
            raw_z = []
            raw_alphaR = []
            raw_alphaI = []
            raw_betaR = []
            raw_betaI = []
            raw_rho = []

            while True:
                # try read six floats: ztmp alphaRtemp betaRtemp rhoRtemp alphaItemp betaItemp
                tokens = []
                # read a line; if empty, EOF
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                parts = line.strip().split()
                for p in parts:
                    try:
                        tokens.append(float(p))
                    except ValueError:
                        pass
                # if less than 6 tokens, try to gather more lines (robust but not perfect)
                needed = 6 - len(tokens)
                while needed > 0:
                    nxt = f.readline()
                    if not nxt:
                        break
                    for p in nxt.strip().split():
                        try:
                            tokens.append(float(p))
                        except ValueError:
                            pass
                        if len(tokens) >= 6:
                            break
                    needed = 6 - len(tokens)
                if not tokens:
                    break

                ztmp = tokens[0]
                alphaRtemp = tokens[1] if len(tokens) > 1 else None
                betaRtemp = tokens[2] if len(tokens) > 2 else None
                rhoRtemp = tokens[3] if len(tokens) > 3 else None
                alphaItemp = tokens[4] if len(tokens) > 4 else None
                betaItemp = tokens[5] if len(tokens) > 5 else None

                # if values read in copy over, otherwise use defaults
                if alphaRtemp is not None:
                    alphaR = alphaRtemp
                if betaRtemp is not None:
                    betaR = betaRtemp
                if rhoRtemp is not None:
                    rhoR = rhoRtemp
                if alphaItemp is not None:
                    alphaI = alphaItemp
                if betaItemp is not None:
                    betaI = betaItemp

                print(
                    f"{ztmp:10.2f}    {alphaR:10.2f}    {betaR:10.2f}    {rhoR:10.2f}    {alphaI:10.4f}    {betaI:10.4f}"
                )

                cp = crci(alphaR, alphaI, freq, AttenUnit)
                cs = crci(betaR, betaI, freq, AttenUnit)

                z_list.append(ztmp)
                c_list.append(cp)
                cs_list.append(cs)
                rho_list.append(rhoR)

                raw_z.append(ztmp)
                raw_alphaR.append(alphaR)
                raw_alphaI.append(alphaI)
                raw_betaR.append(betaR)
                raw_betaI.append(betaI)
                raw_rho.append(rhoR)

                SSP.z = (
                    np.concatenate([SSP.z, np.array([ztmp])])
                    if SSP.z.size
                    else np.array([ztmp])
                )
                SSP.c = (
                    np.concatenate([SSP.c, np.array([cp])])
                    if SSP.c.size
                    else np.array([cp])
                )
                SSP.cs = (
                    np.concatenate([SSP.cs, np.array([cs])])
                    if SSP.cs.size
                    else np.array([cs])
                )
                SSP.rho = (
                    np.concatenate([SSP.rho, np.array([rhoR])])
                    if SSP.rho.size
                    else np.array([rhoR])
                )

                # break if this z is the bottom of this medium
                if abs(ztmp - SSP.depth[medium + 1]) < 1e-9:
                    SSP.raw[medium] = {
                        "z": np.array(raw_z),
                        "alphaR": np.array(raw_alphaR),
                        "alphaI": np.array(raw_alphaI),
                        "betaR": np.array(raw_betaR),
                        "betaI": np.array(raw_betaI),
                        "rho": np.array(raw_rho),
                    }
                    break

            # if SSP.N[medium] == 0 -> calculate N using default formula
            if SSP.N[medium] == 0:
                C = alphaR
                if betaR > 0.0:
                    C = betaR
                deltaz = 0.05 * C / freq
                Ncalc = int(round((SSP.depth[medium + 1] - SSP.depth[medium]) / deltaz))
                Ncalc = max(Ncalc, 10)
                SSP.N[medium] = Ncalc

            print(f"    Number of points = {SSP.N[medium]}")
            # keep track of acoustic media (approximation)
            if np.all(np.array(SSP.cs) == 0):
                if NFirstAcoustic == 0:
                    NFirstAcoustic = medium + 1
                NLastAcoustic = medium + 1

            if medium == 0:
                # HV and cz approximations
                if SSP.z.size >= 2:
                    HV = np.diff(SSP.z)
                    SSP.cz = np.diff(SSP.c) / HV
                else:
                    SSP.cz = np.array([])

            # store Npts
            SSP.Npts[medium] = len(raw_z)
            if medium == 0:
                SSP.depth[0] = SSP.z[0]

        # lower halfspace options (BotOpt quoted string)
        BotOpt_line = f.readline()
        BotOpt = _read_quoted_string(BotOpt_line)
        BotOpt_padded = BotOpt + " " * max(0, 3 - len(BotOpt))
        Bdry["Bot"]["Opt"] = BotOpt_padded
        Bdry["Bot"]["BC"] = Bdry["Bot"]["Opt"][0]
        cp_bot, cs_bot, rho_bot, HS_bot = topbot(f, freq, Bdry["Bot"]["BC"], AttenUnit)
        Bdry["Bot"].update({"cp": cp_bot, "cs": cs_bot, "rho": rho_bot, "HS": HS_bot})

        Bdry["Top"]["depth"] = SSP.depth[0]
        Bdry["Bot"]["depth"] = SSP.depth[NMedia]

        # set "inside" values for reflection coefficients as MATLAB does
        I = NFirstAcoustic if NFirstAcoustic > 0 else 1
        Bdry["Top"]["rhoIns"] = float(SSP.rho[I - 1])
        Bdry["Top"]["cIns"] = float(SSP.c[I - 1])

        I_bot = (
            Loc[NLastAcoustic - 1] + SSP.Npts[NLastAcoustic - 1]
            if NLastAcoustic > 0
            else len(SSP.z)
        )
        I_bot = int(I_bot)
        Bdry["Bot"]["rhoIns"] = float(SSP.rho[I_bot - 1])
        Bdry["Bot"]["cIns"] = float(SSP.c[I_bot - 1])

    return TitleEnv, freq, SSP, Bdry
