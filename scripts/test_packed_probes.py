#!/usr/bin/env python3
"""Compare packed and text probes through the real uniform and AMR solvers.

Build exec/cubedSphereTest.exe first. Requires numpy and h5py.
    python scripts/test_packed_probes.py
    python scripts/test_packed_probes.py --launcher 'mpiexec -n 2'
All simulation files are written to a temporary directory.
"""

import argparse
from datetime import datetime
import importlib.util
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "trajectories_hdf5", ROOT / "scripts" / "trajectories_hdf5.py"
)
generator = importlib.util.module_from_spec(spec)
sys.dont_write_bytecode = True
spec.loader.exec_module(generator)


def run(executable, launcher, directory, amr, error_text=None, overrides=None):
    options = {
        "init_condition_type": 7, "strength": 2, "gamma": 5 / 3,
        "domainSize": 16, "thickness": 16,
        "boxSize_nonrad": 8, "boxSize_rad": 8,
        "inner_radius": 1.4964e12, "outer_radius": 1.61124e13,
        "radial_stretch": 1.5, "spatial_order": 2, "limiter": 1,
        "Riemann_solver_type": 1, "temporal_order": 2, "CFL": 0.1,
        "Sun_gravity": 0, "max_iter": 1, "max_time": 10,
        "convTestType": 0, "boundary_reflux": 1,
        "BC_start_time": 2012.5, "num_CMEs": 0,
        "write_cadence": 1000000, "write_time_cadence": 1e9,
        "slice_cadence": 1000000, "slice_time_cadence": 1e9,
        "checkpoint_cadence": 1000000, "output_divB": 0,
        "amr_max_level": amr, "amr_refine_density_thresh": -1,
        "amr_refine_pressure_thresh": -1, "amr_refine_B_thresh": -1,
        "amr_refine_V_thresh": -1, "amr_refine_regions": "1,4,0,0,8,15,3,11",
        "num_probes": 4, "Probe_cadence": 1,
        "probe0.coords": "trajectories.h5", "probe0.filename": "samples.h5",
        "probe1.coords": "selected.dat", "probe1.filename": "selected_output.dat",
        "probe2.coords": "trajectories.h5", "probe2.filename": "second.h5",
        "probe3.coords": "single.HDF5", "probe3.filename": "single_output.h5",
    }
    options.update(overrides or {})
    inputs = directory / "inputs"
    inputs.write_text("".join(f"-{key} {value}\n" for key, value in options.items()))
    result = subprocess.run(
        [*launcher, str(executable), str(inputs)], cwd=directory,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=180,
    )
    (directory / "run.log").write_text(result.stdout)
    if (result.returncode != 0) != (error_text is not None):
        raise AssertionError(f"Unexpected exit {result.returncode}:\n{result.stdout[-8000:]}")

    if error_text is not None:
        assert error_text in result.stdout, result.stdout[-8000:]
    return result.stdout


def check(executable, launcher, directory, amr):
    directory.mkdir()
    center = np.array([[2012, 183, 1.25, 5, -136.02], [2012, 185, 1.20, 7, -132.02]])
    args = SimpleNamespace(
        lat_min=-8, lat_max=8, lon_min=-8, lon_max=8,
        radial_min_fraction=0.33, radial_max_fraction=1.0, radial_count=3,
        overwrite=False, around="Earth", cme="test", compression="gzip",
        date_start=datetime(2012, 7, 1), date_end=datetime(2012, 7, 3),
        earth_trajectory=directory / "earth.dat",
    )
    generator.write_hdf5(directory / "trajectories.h5", args, center, {})
    with h5py.File(directory / "trajectories.h5") as source:
        selected = source["trajectories"][1, 4, 4]
        names = source["probe_name"][...]
        trajectories = source["trajectories"][...].reshape(-1, 2, 5)
        # A static, one-row trajectory is also valid for packed files.
        single = source["trajectories"][0:1, 0:1, 0:1, 0:1]
    np.savetxt(directory / "selected.dat", selected, fmt="%.12g")
    with h5py.File(directory / "single.HDF5", "w") as file:
        file["trajectories"] = single
    log = run(executable, launcher, directory, amr)
    if amr:
        assert "AMR probe 1 level = 1" in log, "Reference probe must sample a refined cell"
    with h5py.File(directory / "samples.h5") as file:
        # Nine offsets per angle cross both eight-cell reader tile boundaries.
        nsteps = file["time"].shape[0]
        assert nsteps > 0
        assert file["values"].shape == (nsteps, 243, 8)
        assert file["position"].shape == (nsteps, 243, 3)
        np.testing.assert_array_equal(file["probe_name"][...], names)
        assert file.attrs["vector_basis"] == np.bytes_("RTN")
        assert file["position_field_name"].asstr()[...].tolist() == [
            "distance_au", "longitude_deg", "latitude_deg"
        ]
        # The outer radial layer lies beyond the computational shell.
        np.testing.assert_array_equal(file["found"][:, :162], 1)
        np.testing.assert_array_equal(file["found"][:, 162:], 0)
        assert np.isnan(file["values"][:, 162:]).all()
        assert np.isnan(file["position"][:, 162:]).all()
        assert np.isfinite(file["values"][:, :162]).all()
        # Verify every flattened probe, including positions across reader tiles.
        expected = np.empty((nsteps, 162, 3))
        times = file["time"][...]
        for probe, trajectory in enumerate(trajectories[:162]):
            epochs = trajectory[:, 0] + (trajectory[:, 1] - 1) / 366
            expected[:, probe, 0] = np.interp(times, epochs, trajectory[:, 2])
            expected[:, probe, 1] = (np.interp(times, epochs, trajectory[:, 4]) + 181.02) % 360
            expected[:, probe, 2] = np.interp(times, epochs, trajectory[:, 3])
        np.testing.assert_allclose(file["position"][:, :162], expected, rtol=1e-12, atol=1e-12)
        text = np.loadtxt(directory / "selected_output.dat", ndmin=2)
        # Legacy text has four significant digits; packed output retains doubles.
        np.testing.assert_allclose(file["time"][...], text[:, 0], rtol=5e-12)
        np.testing.assert_allclose(file["position"][:, 121, :][:, [0, 2, 1]], text[:, 1:4], rtol=5e-4, atol=1e-8)
        np.testing.assert_allclose(file["values"][:, 121], text[:, 4:], rtol=5e-4, atol=1e-8)
        original = {key: file[key][...] for key in ("time", "position", "values", "found")}
        with h5py.File(directory / "second.h5") as second:
            for key, value in original.items():
                np.testing.assert_equal(second[key][...], value)
    with h5py.File(directory / "single_output.h5") as file:
        np.testing.assert_array_equal(file["found"][...], 1)
        assert np.isfinite(file["values"][...]).all()
    # A subsequent run appends to an existing compatible file.
    run(executable, launcher, directory, amr)
    with h5py.File(directory / "samples.h5") as file:
        assert file["time"].shape == (2 * nsteps,)
        for key, value in original.items():
            np.testing.assert_equal(file[key][:nsteps], value)
    # Same shape but different coordinates must fail without adding samples.
    with h5py.File(directory / "trajectories.h5", "r+") as file:
        file["trajectories"][0, 0, 0, 0, 4] += 1
    run(executable, launcher, directory, amr, error_text="incompatible with these trajectories")
    with h5py.File(directory / "samples.h5") as file:
        assert file["time"].shape == (2 * nsteps,)
    # Configuration aliases must fail before any trajectory input is opened for writing.
    before = (directory / "trajectories.h5").read_bytes()
    run(executable, launcher, directory, amr,
        error_text="Probe output must differ from every trajectory input",
        overrides={"probe0.filename": "./trajectories.h5"})
    assert (directory / "trajectories.h5").read_bytes() == before
    run(executable, launcher, directory, amr,
        error_text="Each packed probe set requires a distinct output file",
        overrides={"probe2.filename": "./samples.h5"})
    print(f"{'AMR' if amr else 'Uniform'} packed probes: passed", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, default=ROOT / "exec" / "cubedSphereTest.exe")
    parser.add_argument("--launcher", default="", help="MPI launcher and arguments")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="heliocubed-probes-") as temporary:
        for amr in (0, 1):
            check(args.executable.resolve(), shlex.split(args.launcher), Path(temporary) / str(amr), amr)


if __name__ == "__main__":
    main()
