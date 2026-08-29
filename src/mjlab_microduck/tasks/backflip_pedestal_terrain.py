"""Flat floor with a centered launch cube for backflip curriculum training."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab.terrains.terrain_generator import (
    SubTerrainCfg,
    TerrainGeometry,
    TerrainOutput,
)


# A literal 25 cm cube.  The earlier 18 cm top left the nominal MicroDuck
# stance on the edges, so tiny policy motions caused an accidental fall before
# the launch phase.  This size preserves the intended elevated-start benefit
# while leaving room for a deliberate push toward the lower floor.
PEDESTAL_HEIGHT = 0.25
PEDESTAL_WIDTH = 0.25


@dataclass(kw_only=True)
class BackflipPedestalTerrainCfg(SubTerrainCfg):
    """A real collision cube above a lower, continuous landing floor."""

    pedestal_height: float = PEDESTAL_HEIGHT
    pedestal_width: float = PEDESTAL_WIDTH
    floor_thickness: float = 0.10

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        del difficulty, rng
        if self.pedestal_width >= min(self.size):
            raise ValueError("pedestal must be smaller than its terrain tile")
        body = spec.body("terrain")
        center_x = self.size[0] / 2.0
        center_y = self.size[1] / 2.0

        floor = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(
                self.size[0] / 2.0,
                self.size[1] / 2.0,
                self.floor_thickness / 2.0,
            ),
            pos=(center_x, center_y, -self.floor_thickness / 2.0),
        )
        pedestal = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(
                self.pedestal_width / 2.0,
                self.pedestal_width / 2.0,
                self.pedestal_height / 2.0,
            ),
            pos=(center_x, center_y, self.pedestal_height / 2.0),
        )
        # Keep the environment origin on the lower floor. Reset heights and
        # landing gates can then distinguish the elevated start from a genuine
        # lower-floor landing without relying on geom-name ordering.
        origin = np.array([center_x, center_y, 0.0])
        return TerrainOutput(
            origin=origin,
            geometries=[
                TerrainGeometry(geom=floor, color=(0.32, 0.38, 0.45, 1.0)),
                TerrainGeometry(geom=pedestal, color=(0.85, 0.55, 0.12, 1.0)),
            ],
        )
