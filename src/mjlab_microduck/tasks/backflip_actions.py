"""Action terms for the assist-to-autonomy backflip curriculum."""

from dataclasses import dataclass

import torch
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg


@dataclass(kw_only=True)
class BackflipResidualJointPositionActionCfg(JointPositionActionCfg):
    """Joint targets with residual authority coupled to spotter strength."""

    min_assisted_authority: float = 0.05
    full_authority_after_angle_rad: float | None = None

    def build(self, env):
        return BackflipResidualJointPositionAction(self, env)


class BackflipResidualJointPositionAction(JointPositionAction):
    """Interpolate policy targets toward nominal PD while fully assisted.

    At assist scale one, the actor has only ``min_assisted_authority`` of its
    normal target displacement. At assist scale zero this is exactly the stock
    action term, so strict evaluation and eventual autonomous training retain
    full control authority.
    """

    cfg: BackflipResidualJointPositionActionCfg

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)
        assist = min(
            max(float(getattr(self._env, "_backflip_assist_scale", 0.0)), 0.0), 1.0
        )
        minimum = min(max(float(self.cfg.min_assisted_authority), 0.0), 1.0)
        eligible = getattr(self._env, "_backflip_assist_eligible", None)
        if eligible is None:
            effective_assist = torch.full(
                (self.num_envs, 1), assist, device=self.device
            )
        else:
            # Reverse-curriculum worlds receive no wrench and need full action
            # authority to learn the landing/recovery half of the maneuver.
            constrained = eligible
            flight_ended = getattr(self._env, "_backflip_flight_ended_latch", None)
            if flight_ended is not None:
                # Preserve the validated teacher through launch and flight, but
                # restore full authority on the control step after recontact so
                # the actor can absorb impact and stand.
                constrained = constrained & ~flight_ended
            release_angle = self.cfg.full_authority_after_angle_rad
            if release_angle is not None:
                progress = getattr(self._env, "_backflip_max", None)
                if progress is not None:
                    # Preserve the nominal-PD teacher through launch and tuck,
                    # then let the actor actively extend and brake before
                    # impact. The virtual-spotter pulse has ended well before
                    # this phase, so releasing action authority does not add
                    # external energy to the touchdown controller.
                    constrained = constrained & (progress < float(release_angle))
            effective_assist = constrained.to(self._processed_actions.dtype).unsqueeze(1)
            effective_assist = effective_assist * assist
        authority = minimum + (1.0 - minimum) * (1.0 - effective_assist)
        self._processed_actions = self._offset + authority * (
            self._processed_actions - self._offset
        )
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )
