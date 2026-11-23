import pytest
import graphics.contact_shadow_reproject as mod

CASES = [
    ("reproject_matrix", (1,)),
    ("thickness_step", (1,)),
    ("queue_fence", (1,)),
    ("step_world_space", (1,)),
    ("teleport_invalidate", (1,)),
    ("determinism", (1,)),
    ("history_clip", (1,)),
    ("validation", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("half_res", (1,)),
    ("metrics", (1,)),
    ("ray_march", (1,)),
    ("depth_bias", (1,)),
    ("softness", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
