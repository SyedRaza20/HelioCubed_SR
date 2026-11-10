#!/usr/bin/env python3
"""Create the probe trajectories from ``trajectories.py`` in one HDF5 file.

The main dataset has the layout

    trajectories[radius, latitude_offset, longitude_offset, time, field]

where ``field`` is ``year``, ``day_of_year``, ``distance_au``,
``latitude_deg``, or ``longitude_deg``.  Axis values and the legacy ``.dat``
filenames are stored in the same file, so any old trajectory can be recovered
without creating thousands of small files.

Example (equivalent to the CME example in trajectories.py):
    python trajectories_hdf5.py -30 30 -30 30 07/10/2012 07/20/2012 --around CME --overwrite
    python trajectories_hdf5.py -30 30 -30 30 04/01/2010 10/15/2012 --around Earth --overwrite

The default output is (subject to change):
    (1) For cme trajectories: "/anvil/projects/x-mca07s033/syedraza/15CMEs_Syed/CMEs_nowcast" + args.cme + "trajectories.h5"
    (2) For earth trajectories: "/anvil/projects/x-mca07s033/syedraza/15CMEs_Syed/CMEs_nowcast/trajectories.h5"

Each CME will have its own CME trajectories, but the earth one can be shared. Use ``--output`` to select another path.
"""

import argparse
import os, sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import h5py
import numpy as np


FIELD_NAMES = (
    "year",
    "day_of_year",
    "distance_au",
    "latitude_deg",
    "longitude_deg",
)
FORMAT_NAME = "MS-FLUKSS probe trajectories"
FORMAT_VERSION = 1

syedraza_dir = "/anvil/projects/x-mca07s033/syedraza"

def parse_date(value):
    """Parse the MM/DD/YYYY dates accepted by the original script."""
    try:
        return datetime.strptime(value, "%m/%d/%Y")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "dates must use MM/DD/YYYY format: {!r}".format(value)
        ) from exc


def parse_args():
    script_dir = Path(__file__).resolve().parent
    syedraza_dir = script_dir.parent.parent

    parser = argparse.ArgumentParser(
        description=(
            "Generate the regular MS-FLUKSS probe grid as one compressed HDF5 "
            "file instead of one .dat file per probe."
        )
    )
    parser.add_argument("lon_min", type=int, help="minimum longitude offset in degrees")
    parser.add_argument("lon_max", type=int, help="maximum longitude offset in degrees")
    parser.add_argument("lat_min", type=int, help="minimum latitude offset in degrees")
    parser.add_argument("lat_max", type=int, help="maximum latitude offset in degrees")
    parser.add_argument("date_start", type=parse_date, help="first date, MM/DD/YYYY")
    parser.add_argument("date_end", type=parse_date, help="last date, MM/DD/YYYY")
    parser.add_argument(
        "--around",
        choices=("CME", "Earth"),
        default="CME",
        help="center the probe grid on the CME or Earth (default: CME)",
    )
    parser.add_argument(
        "--cme",
        default="11_2012-07-12",
        help="CME directory name (default: 11_2012-07-12)",
    )
    parser.add_argument(
        "--earth-trajectory",
        type=Path,
        default=syedraza_dir / "CMEtests" / "CME11" / "trajEarth.dat",
        help="Earth trajectory used for Earth probes and the HGI transformation",
    )
    parser.add_argument(
        "--gcs-file",
        type=Path,
        default=None,
        help="GCS input file; defaults to ../CMEs_nowcast/CME/GCS.txt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output HDF5 path (a mode-specific default is used if omitted)",
    )
    parser.add_argument(
        "--radial-count",
        type=int,
        default=20,
        help="number of radial probe layers (default: 20)",
    )
    parser.add_argument(
        "--radial-min-fraction",
        type=float,
        default=0.33,
        help="innermost fraction of the center distance (default: 0.33)",
    )
    parser.add_argument(
        "--radial-max-fraction",
        type=float,
        default=1.0,
        help="outermost fraction of the center distance (default: 1.0)",
    )
    parser.add_argument(
        "--cme-distance-au",
        type=float,
        default=1.25,
        help="CME-centered cone distance in AU (default: 1.25)",
    )
    parser.add_argument(
        "--compression",
        choices=("gzip", "lzf", "none"),
        default="gzip",
        help="HDF5 compression filter (default: gzip)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing output file",
    )

    args = parser.parse_args()

    if args.gcs_file is None:
        args.gcs_file = script_dir.parent / "CMEs_nowcast" / args.cme / "GCS.txt"
    if args.output is None:
        if args.around == "CME":
            args.output = syedraza_dir / "15CMEs_Syed" / "CMEs_nowcast" / args.cme / "trajectories.h5"
        else:
            args.output = syedraza_dir / "15CMEs_Syed" / "CMEs_nowcast" / "earth_trajectories.h5"

    if args.lon_min > args.lon_max or args.lat_min > args.lat_max:
        parser.error("minimum offsets must not exceed maximum offsets")
    if args.date_start > args.date_end:
        parser.error("date_start must not be later than date_end")
    if args.radial_count <= 0:
        parser.error("--radial-count must be positive")
    if args.radial_min_fraction <= 0:
        parser.error("--radial-min-fraction must be positive")
    if args.radial_min_fraction > args.radial_max_fraction:
        parser.error("minimum radial fraction must not exceed maximum radial fraction")
    if args.cme_distance_au <= 0:
        parser.error("--cme-distance-au must be positive")

    return args


def even_offsets(minimum, maximum):
    """Return the even integer offsets selected by trajectories.py."""
    values = np.arange(minimum, maximum + 1, dtype=np.int16)
    values = values[values % 2 == 0]
    if values.size == 0:
        raise ValueError("the requested interval contains no even-degree offsets")
    return values


def load_earth_trajectory(path):
    data = np.loadtxt(path, dtype=np.float64, ndmin=2)
    if data.shape[1] != 5:
        raise ValueError(
            "{} must have five columns: year, DOY, distance, latitude, longitude".format(
                path
            )
        )
    return data


def date_code(dt):
    """Make an integer that preserves chronological order for daily data."""
    return dt.year * 1000 + int(dt.strftime("%j"))


def year_fraction(dt):
    start = datetime(dt.year, 1, 1)
    end = datetime(dt.year + 1, 1, 1)
    return dt.year + (dt - start).total_seconds() / (end - start).total_seconds()


def earth_center_trajectory(earth_data, start, end):
    codes = earth_data[:, 0].astype(np.int64) * 1000 + earth_data[:, 1].astype(
        np.int64
    )
    selected = earth_data[(codes >= date_code(start)) & (codes <= date_code(end))]
    if selected.size == 0:
        raise ValueError(
            "the Earth trajectory has no rows from {} through {}".format(
                start.strftime("%m/%d/%Y"), end.strftime("%m/%d/%Y")
            )
        )
    return selected


def cme_center_trajectory(earth_data, gcs_file, start, end, distance_au):
    gcs = np.genfromtxt(gcs_file, dtype=str, ndmin=2)
    if gcs.shape[1] < 6:
        raise ValueError("{} must contain at least six columns".format(gcs_file))

    final_fit_time = datetime.fromisoformat(gcs[-1, 0].replace("/", "-"))
    cme_latitude = float(gcs[-1, 4])
    cme_stonyhurst_longitude = float(gcs[-1, 5])

    earth_years = earth_data[:, 0].astype(np.int64)
    earth_doys = earth_data[:, 1]
    days_per_year = np.where(
        (earth_years % 400 == 0)
        | ((earth_years % 100 != 0) & (earth_years % 4 == 0)),
        366.0,
        365.0,
    )
    earth_times = earth_years + (earth_doys - 1.0) / days_per_year
    earth_hgi_longitude = float(
        np.interp(year_fraction(final_fit_time), earth_times, earth_data[:, 4])
    )
    cme_hgi_longitude = earth_hgi_longitude + cme_stonyhurst_longitude

    count = (end - start).days + 1
    dates = [start + timedelta(days=offset) for offset in range(count)]
    trajectory = np.empty((count, 5), dtype=np.float64)
    trajectory[:, 0] = [date.year for date in dates]
    trajectory[:, 1] = [int(date.strftime("%j")) for date in dates]
    trajectory[:, 2] = distance_au
    trajectory[:, 3] = cme_latitude
    trajectory[:, 4] = cme_hgi_longitude

    metadata = {
        "gcs_final_fit_time": final_fit_time.isoformat(timespec="minutes"),
        "cme_latitude_deg": cme_latitude,
        "cme_stonyhurst_longitude_deg": cme_stonyhurst_longitude,
        "earth_hgi_longitude_deg": earth_hgi_longitude,
        "cme_hgi_longitude_deg": cme_hgi_longitude,
    }
    return trajectory, metadata


def probe_names(radius_count, latitude_offsets, longitude_offsets):
    names = np.empty(
        (radius_count, latitude_offsets.size, longitude_offsets.size), dtype=object
    )
    for radius_index in range(radius_count):
        for latitude_index, latitude_offset in enumerate(latitude_offsets):
            for longitude_index, longitude_offset in enumerate(longitude_offsets):
                names[radius_index, latitude_index, longitude_index] = (
                    "probe_r{}_lat{}_lon{}.dat".format(
                        radius_index, int(latitude_offset), int(longitude_offset)
                    )
                )
    return names


def compression_options(name):
    if name == "none":
        return {}
    options = {"compression": name, "shuffle": True}
    if name == "gzip":
        options["compression_opts"] = 4
    return options


def write_hdf5(output, args, center, metadata):
    latitude_offsets = even_offsets(args.lat_min, args.lat_max)
    longitude_offsets = even_offsets(args.lon_min, args.lon_max)
    radius_fractions = np.linspace(
        args.radial_min_fraction,
        args.radial_max_fraction,
        args.radial_count,
        dtype=np.float64,
    )
    names = probe_names(args.radial_count, latitude_offsets, longitude_offsets)

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.overwrite:
        raise FileExistsError(
            "{} already exists; pass --overwrite to replace it".format(output)
        )

    temporary = tempfile.NamedTemporaryFile(
        prefix=".{}-".format(output.name),
        suffix=".tmp",
        dir=str(output.parent),
        delete=False,
    )
    temporary_path = Path(temporary.name)
    temporary.close()

    time_count = center.shape[0]
    chunks = (
        1,
        min(8, latitude_offsets.size),
        min(8, longitude_offsets.size),
        min(128, time_count),
        len(FIELD_NAMES),
    )

    try:
        with h5py.File(temporary_path, "w") as h5:
            h5.attrs["format_name"] = FORMAT_NAME
            h5.attrs["format_version"] = FORMAT_VERSION
            h5.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
            h5.attrs["generator"] = Path(__file__).name
            h5.attrs["center"] = args.around
            h5.attrs["cme"] = args.cme
            h5.attrs["date_start"] = args.date_start.strftime("%Y-%m-%d")
            h5.attrs["date_end"] = args.date_end.strftime("%Y-%m-%d")
            h5.attrs["probe_count"] = int(names.size)
            h5.attrs["axis_order"] = (
                "radius, latitude_offset, longitude_offset, time, field"
            )
            h5.attrs["legacy_row_format"] = (
                "year day_of_year distance_au latitude_deg longitude_deg"
            )
            h5.attrs["earth_trajectory_source"] = str(
                args.earth_trajectory.resolve()
            )
            if args.around == "CME":
                h5.attrs["gcs_source"] = str(args.gcs_file.resolve())
            for key, value in metadata.items():
                h5.attrs[key] = value

            axes = h5.create_group("axes")
            axes.create_dataset("radius_fraction", data=radius_fractions)
            axes.create_dataset("latitude_offset_deg", data=latitude_offsets)
            axes.create_dataset("longitude_offset_deg", data=longitude_offsets)
            axes.create_dataset(
                "field_name",
                data=np.asarray(FIELD_NAMES, dtype=h5py.string_dtype("utf-8")),
            )

            time_group = h5.create_group("time")
            years = center[:, 0].astype(np.int32)
            days_of_year = center[:, 1].astype(np.int16)
            time_group.create_dataset("year", data=years)
            time_group.create_dataset("day_of_year", data=days_of_year)
            iso_dates = [
                (datetime(int(year), 1, 1) + timedelta(days=int(day) - 1)).strftime(
                    "%Y-%m-%d"
                )
                for year, day in zip(years, days_of_year)
            ]
            time_group.create_dataset(
                "iso_date",
                data=np.asarray(iso_dates, dtype=h5py.string_dtype("utf-8")),
            )

            h5.create_dataset(
                "probe_name",
                data=names,
                dtype=h5py.string_dtype("utf-8"),
                **compression_options(args.compression)
            )

            dataset = h5.create_dataset(
                "trajectories",
                shape=(
                    args.radial_count,
                    latitude_offsets.size,
                    longitude_offsets.size,
                    time_count,
                    len(FIELD_NAMES),
                ),
                dtype=np.float64,
                chunks=chunks,
                **compression_options(args.compression)
            )
            dataset.attrs["field_names"] = np.asarray(
                FIELD_NAMES, dtype=h5py.string_dtype("utf-8")
            )
            dataset.attrs["units"] = np.asarray(
                ("year", "day", "AU", "degree", "degree"),
                dtype=h5py.string_dtype("utf-8"),
            )

            # Write one radial layer at a time so long date ranges do not require
            # constructing the complete five-dimensional array in memory.
            for radius_index, radius_fraction in enumerate(radius_fractions):
                block = np.empty(
                    (
                        latitude_offsets.size,
                        longitude_offsets.size,
                        time_count,
                        len(FIELD_NAMES),
                    ),
                    dtype=np.float64,
                )
                block[..., 0] = years[None, None, :]
                block[..., 1] = days_of_year[None, None, :]
                block[..., 2] = np.round(
                    center[:, 2] * radius_fraction, 4
                )[None, None, :]
                block[..., 3] = np.round(
                    center[:, 3][None, None, :]
                    + latitude_offsets[:, None, None],
                    4,
                )
                block[..., 4] = np.round(
                    center[:, 4][None, None, :]
                    + longitude_offsets[None, :, None],
                    4,
                )
                dataset[radius_index, ...] = block

            h5.flush()

        # NamedTemporaryFile starts with mode 0600.  Restore the permissions a
        # normal newly-created output would receive from the caller's umask so
        # project collaborators can read the finished file when appropriate.
        current_umask = os.umask(0)
        os.umask(current_umask)
        os.chmod(temporary_path, 0o666 & ~current_umask)
        os.replace(temporary_path, output)
    except BaseException:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise

    return output, names.size, time_count


def main():
    args = parse_args()
    earth_data = load_earth_trajectory(args.earth_trajectory)

    if args.around == "Earth":
        center = earth_center_trajectory(
            earth_data, args.date_start, args.date_end
        )
        metadata = {}
    else:
        center, metadata = cme_center_trajectory(
            earth_data,
            args.gcs_file,
            args.date_start,
            args.date_end,
            args.cme_distance_au,
        )

    output, probe_count, time_count = write_hdf5(
        args.output, args, center, metadata
    )
    print("Wrote {}".format(output))
    print("Probes: {:,}".format(probe_count))
    print("Times per probe: {:,}".format(time_count))
    print("Dataset: trajectories[radius, latitude_offset, longitude_offset, time, field]")


if __name__ == "__main__":
    main()
