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
LANDING_MAT_HEIGHT = 0.18


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


@dataclass(kw_only=True)
class BackflipMatTerrainCfg(SubTerrainCfg):
    """Launch cube surrounded by a raised, compliant gymnastics mat.

    The mat is an explicit curriculum aid: its top is 7 cm below the cube and
    its MuJoCo contact time constant is softer than the rigid lower floor.
    Later stages can lower and stiffen it without changing the robot or policy
    interface.
    """

    pedestal_height: float = PEDESTAL_HEIGHT
    pedestal_width: float = PEDESTAL_WIDTH
    landing_mat_height: float = LANDING_MAT_HEIGHT
    floor_thickness: float = 0.10
    mat_contact_time_s: float = 0.06
    mat_contact_damping: float = 1.0
    mat_slide_friction: float = 1.0

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        del difficulty, rng
        if not 0.0 < self.landing_mat_height < self.pedestal_height:
            raise ValueError("landing mat must be above floor and below cube top")
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
        mat = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(
                self.size[0] / 2.0,
                self.size[1] / 2.0,
                self.landing_mat_height / 2.0,
            ),
            pos=(center_x, center_y, self.landing_mat_height / 2.0),
            friction=(self.mat_slide_friction, 0.005, 0.0001),
            solref=(self.mat_contact_time_s, self.mat_contact_damping),
            solimp=(0.85, 0.95, 0.001, 0.5, 2.0),
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
        origin = np.array([center_x, center_y, 0.0])
        return TerrainOutput(
            origin=origin,
            geometries=[
                TerrainGeometry(geom=floor, color=(0.20, 0.24, 0.29, 1.0)),
                TerrainGeometry(geom=mat, color=(0.18, 0.50, 0.72, 1.0)),
                TerrainGeometry(geom=pedestal, color=(0.85, 0.55, 0.12, 1.0)),
            ],
        )
