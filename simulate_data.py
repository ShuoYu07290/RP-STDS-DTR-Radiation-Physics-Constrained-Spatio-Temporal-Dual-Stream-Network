"""Coarse infrared sequence simulator supplied with the code excerpt.

This simplified generator preserves only the physical chain described in the
paper: class-dependent micro-motion -> projected area -> lumped heat balance ->
band-integrated radiation -> small-target image formation. Calibrated geometry,
material tables, orbit files and the sensor model are not included.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image


PLANCK_H = 6.62607015e-34
LIGHT_C = 2.99792458e8
BOLTZMANN_K = 1.380649e-23
STEFAN_BOLTZMANN = 5.670374419e-8

FRAMES = 100
TIME_STEP = 1.0
IMAGE_SIZE = 256
DETECTOR_DIRECTION = np.asarray([0.0, 0.0, 1.0])

# Deliberately coarse representative parameters, not the calibrated experiment.
CLASS_PARAMETERS: Dict[str, Dict[str, float]] = {
    "target": {
        "spin": 0.035,
        "precession": 0.010,
        "emissivity": 0.75,
        "heat_capacity": 1.8e5,
        "area_scale": 1.00,
    },
    "decoy": {
        "spin": 0.090,
        "precession": 0.025,
        "emissivity": 0.62,
        "heat_capacity": 5.5e4,
        "area_scale": 0.72,
    },
    "debris": {
        "spin": 0.180,
        "precession": 0.055,
        "emissivity": 0.85,
        "heat_capacity": 2.5e4,
        "area_scale": 0.45,
    },
}


def rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    skew = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    identity = np.eye(3)
    return identity + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def attitude(kind: str, time: float, phase: float) -> np.ndarray:
    parameter = CLASS_PARAMETERS[kind]
    spin = rotation_matrix(np.asarray([0.2, 0.6, 0.77]),
                           parameter["spin"] * time + phase)
    precession = rotation_matrix(np.asarray([0.0, 0.0, 1.0]),
                                 parameter["precession"] * time)
    return precession @ spin


def projected_area(kind: str, rotation: np.ndarray, direction: np.ndarray) -> float:
    body_normal = rotation @ np.asarray([0.6135, 0.3519, 0.7068])
    visibility = abs(float(np.dot(body_normal, direction)))
    return CLASS_PARAMETERS[kind]["area_scale"] * max(visibility, 1e-4)


def band_radiance(temperature: float) -> float:
    """Integrate Planck radiance over the 3.7--4.8 micrometre band."""
    wavelength = np.linspace(3.7e-6, 4.8e-6, 64)
    numerator = 2.0 * PLANCK_H * LIGHT_C**2
    exponent = PLANCK_H * LIGHT_C / (wavelength * BOLTZMANN_K * temperature)
    spectral = numerator / (wavelength**5 * np.expm1(exponent))
    return float(np.trapz(spectral, wavelength))


def simulate_radiation(kind: str, rng: np.random.Generator) -> np.ndarray:
    parameter = CLASS_PARAMETERS[kind]
    sun_direction = np.asarray([0.70, 0.20, 0.685])
    sun_direction /= np.linalg.norm(sun_direction)
    earth_direction = np.asarray([-0.30, 0.40, 0.866])
    earth_direction /= np.linalg.norm(earth_direction)

    temperature = 285.0 + rng.normal(0.0, 2.0)
    phase = rng.uniform(0.0, 2.0 * math.pi)
    sequence = []

    for frame_index in range(FRAMES):
        time = frame_index * TIME_STEP
        rotation = attitude(kind, time, phase)
        visible_area = projected_area(kind, rotation, DETECTOR_DIRECTION)
        solar_area = projected_area(kind, rotation, sun_direction)
        earth_area = projected_area(kind, rotation, earth_direction)

        absorbed_heat = 1362.0 * 0.65 * solar_area + 237.0 * earth_area + 20.0
        emitted_heat = (parameter["emissivity"] * STEFAN_BOLTZMANN
                        * visible_area * temperature**4)
        temperature += TIME_STEP * (absorbed_heat - emitted_heat) \
            / parameter["heat_capacity"]

        received = (0.85 * parameter["emissivity"] * visible_area
                    * band_radiance(temperature))
        sequence.append(received)

    sequence = np.asarray(sequence, dtype=np.float32)
    return (sequence - sequence.min()) / (np.ptp(sequence) + 1e-8)


def render_sequence(
    radiation: np.ndarray,
    output_directory: Path,
    rng: np.random.Generator,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    for index, value in enumerate(radiation):
        image = rng.normal(35.0, 8.0, size=(IMAGE_SIZE, IMAGE_SIZE))
        x = int(35 + 1.8 * index + 0.20 * index**2 / FRAMES) % IMAGE_SIZE
        y = int(128 + 45 * math.sin(2.0 * math.pi * index / FRAMES))
        amplitude = 45.0 + 150.0 * float(value)
        image[max(0, y - 1):min(IMAGE_SIZE, y + 2),
              max(0, x - 1):min(IMAGE_SIZE, x + 2)] += amplitude
        Image.fromarray(np.clip(image, 0, 255).astype(np.uint8)).save(
            output_directory / f"frame_{index:05d}.png"
        )


def generate(output_root: Path, samples_per_class: int, seed: int) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for class_name in CLASS_PARAMETERS:
        for sample_index in range(samples_per_class):
            sample_root = output_root / class_name / f"sample_{sample_index:04d}"
            radiation = simulate_radiation(class_name, rng)
            sample_root.mkdir(parents=True, exist_ok=True)
            np.save(sample_root / "radiation.npy", radiation)
            render_sequence(radiation, sample_root / "frames", rng)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Coarse infrared data simulator")
    parser.add_argument("--output", type=Path, default=Path("rough_simulated_data"))
    parser.add_argument("--samples-per-class", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate(args.output, args.samples_per_class, args.seed)


# Omitted: high-fidelity shape mesh, material database, calibrated orbit/attitude
# histories, atmospheric propagation, detector MTF, nonuniformity and noise model.
